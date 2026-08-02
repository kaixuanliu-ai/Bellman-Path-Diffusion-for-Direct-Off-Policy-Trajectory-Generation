"""The single Bellman diffusion objective from Equation (48).

This module uses the epsilon-prediction parameterization stated in Appendix C
of the paper.  Branch selection realizes the mixture weights ``1-gamma`` and
``gamma``; component targets themselves are never multiplied by ``gamma``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from bpd.core.branches import ContinueBranch, StopBranch, TeacherNoiseFn
from bpd.data.replay import SuffixReplayBuffer
from bpd.models.diffusion import BlockwiseDiffusion, DDPMSchedule
from bpd.utils.arrays import DataBatch


def get_loss_weight(
    t: Tensor,
    schedule: DDPMSchedule,
    mode: str = "uniform",
    min_snr_gamma: float = 5.0,
) -> Tensor:
    """Return per-example diffusion loss weights.

    ``uniform`` is the paper's default discrete DDPM objective.  The other
    modes are explicit optional variants and do not alter Bellman branching.
    """
    if mode == "uniform":
        return torch.ones(t.shape[0], dtype=torch.float32, device=t.device)
    alpha_bar = schedule.alphas_bar.to(t.device)[t]
    snr = alpha_bar / (1.0 - alpha_bar).clamp_min(1e-8)
    if mode == "snr":
        return snr.float()
    if mode == "min_snr_gamma":
        if min_snr_gamma <= 0:
            raise ValueError("min_snr_gamma must be positive")
        return snr.clamp(max=min_snr_gamma).float()
    raise ValueError(f"unknown loss weighting mode: {mode!r}")


@dataclass(frozen=True)
class BellmanTargetBatch:
    """Materialized branch samples used by one objective evaluation."""

    z_t: Tensor
    target_noise: Tensor
    conditioning: Tensor
    successor_conditioning: Tensor
    timesteps: Tensor
    continuation: Tensor


class BellmanDiffusionLoss(nn.Module):
    """Sample and evaluate the Bellman mixture objective.

    The evaluation-policy action ``next_action`` is required explicitly.  A
    logged behavior action is not a valid substitute for
    ``a' ~ pi_e(. | s')`` (Equation (3)).
    """

    def __init__(
        self,
        gamma: float,
        schedule: DDPMSchedule,
        token_dim: int,
        loss_weight_fn: Optional[Callable[[Tensor, DDPMSchedule], Tensor]] = None,
        diffusion: Optional[BlockwiseDiffusion] = None,
    ) -> None:
        super().__init__()
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must lie in (0, 1), got {gamma}")
        self.gamma = float(gamma)
        self.schedule = schedule
        self.token_dim = int(token_dim)
        self.loss_weight_fn = loss_weight_fn
        self.diffusion = diffusion or BlockwiseDiffusion(schedule, token_dim)
        if self.diffusion.token_dim != token_dim or self.diffusion.T != schedule.T:
            raise ValueError("diffusion is incompatible with schedule/token_dim")
        self.stop_branch = StopBranch(self.diffusion)
        self.continue_branch = ContinueBranch(self.diffusion)

    def forward(
        self,
        noise_net: nn.Module,
        teacher_noise_fn: Optional[TeacherNoiseFn],
        batch: DataBatch,
        h: int,
        *,
        next_action: Tensor,
        replay_buffer: Optional[SuffixReplayBuffer] = None,
    ) -> Tensor:
        targets = self.build_targets(
            teacher_noise_fn=teacher_noise_fn,
            batch=batch,
            h=h,
            next_action=next_action,
            replay_buffer=replay_buffer,
        )
        prediction = noise_net(
            targets.z_t,
            targets.timesteps,
            targets.conditioning,
            h,
        )
        if prediction.shape != targets.target_noise.shape:
            raise ValueError(
                f"student returned {tuple(prediction.shape)}, expected "
                f"{tuple(targets.target_noise.shape)}"
            )
        per_example = (prediction - targets.target_noise).square().mean(dim=(1, 2))
        if self.loss_weight_fn is None:
            weights = torch.ones_like(per_example)
        else:
            weights = self.loss_weight_fn(targets.timesteps, self.schedule).to(
                device=per_example.device, dtype=per_example.dtype
            )
        return (weights * per_example).mean()

    def build_targets(
        self,
        teacher_noise_fn: Optional[TeacherNoiseFn],
        batch: DataBatch,
        h: int,
        *,
        next_action: Tensor,
        replay_buffer: Optional[SuffixReplayBuffer] = None,
        timesteps: Optional[Tensor] = None,
        continuation: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> BellmanTargetBatch:
        """Construct Algorithm 1's sampled branch and Appendix C target.

        Optional ``timesteps``, ``continuation``, and ``noise`` arguments make
        the mathematical target directly testable without changing production
        sampling behavior.
        """
        if h < 1:
            raise ValueError(f"h must be positive, got {h}")
        state = batch.state
        action = batch.action
        reward = batch.reward.reshape(-1, 1)
        next_state = batch.next_state
        batch_size = state.shape[0]
        if next_action.shape[0] != batch_size:
            raise ValueError("next_action batch size does not match transitions")

        next_action = next_action.to(device=state.device, dtype=state.dtype)
        y = torch.cat((reward.to(state), next_state.to(state), next_action), dim=-1)
        if y.shape[-1] != self.token_dim:
            raise ValueError(
                f"token has dimension {y.shape[-1]}, expected {self.token_dim}"
            )
        x = torch.cat((state, action), dim=-1)
        x_prime = torch.cat((next_state, next_action), dim=-1)

        if timesteps is None:
            timesteps = torch.randint(
                1, self.schedule.T + 1, (batch_size,), device=state.device
            )
        else:
            timesteps = timesteps.to(device=state.device, dtype=torch.long)
        if continuation is None:
            continuation = torch.rand(batch_size, device=state.device) < self.gamma
        else:
            continuation = continuation.to(device=state.device, dtype=torch.bool)
        if h == 1:
            continuation = torch.zeros_like(continuation)

        expected_shape = (batch_size, h, self.token_dim)
        if noise is None:
            noise = torch.randn(expected_shape, device=state.device, dtype=state.dtype)
        elif noise.shape != expected_shape:
            raise ValueError(f"noise must have shape {expected_shape}")

        z_t = torch.empty_like(noise)
        target_noise = torch.empty_like(noise)

        stop_mask = ~continuation
        if stop_mask.any():
            stop = self.stop_branch.build(
                y[stop_mask],
                timesteps[stop_mask],
                h,
                noise=noise[stop_mask],
            )
            z_t[stop_mask] = stop.z_t
            target_noise[stop_mask] = stop.target_noise

        if continuation.any():
            if teacher_noise_fn is None:
                raise ValueError("teacher_noise_fn is required for h > 1 continuation")
            cont_indices = continuation.nonzero(as_tuple=False).squeeze(-1)
            cont_x_prime = x_prime[cont_indices]
            clean_suffix = self._clean_suffixes(
                teacher_noise_fn,
                cont_x_prime,
                h - 1,
                replay_buffer,
            )
            cont = self.continue_branch.build(
                y[cont_indices],
                cont_x_prime,
                timesteps[cont_indices],
                h,
                clean_suffix,
                teacher_noise_fn,
                noise=noise[cont_indices],
            )
            z_t[cont_indices] = cont.z_t
            target_noise[cont_indices] = cont.target_noise

        return BellmanTargetBatch(
            z_t=z_t,
            target_noise=target_noise,
            conditioning=x,
            successor_conditioning=x_prime,
            timesteps=timesteps,
            continuation=continuation,
        )

    def _clean_suffixes(
        self,
        teacher_noise_fn: TeacherNoiseFn,
        x_prime: Tensor,
        suffix_horizon: int,
        replay_buffer: Optional[SuffixReplayBuffer],
    ) -> Tensor:
        """Retrieve exact-key suffixes and generate cache misses in one batch."""
        batch_size = x_prime.shape[0]
        result = x_prime.new_empty(batch_size, suffix_horizon, self.token_dim)
        missing: list[int] = []
        for index in range(batch_size):
            cached = (
                replay_buffer.sample_or_none(x_prime[index])
                if replay_buffer is not None
                else None
            )
            if cached is None:
                missing.append(index)
            else:
                result[index] = cached.to(device=x_prime.device, dtype=x_prime.dtype)

        if missing:
            missing_index = torch.tensor(
                missing, device=x_prime.device, dtype=torch.long
            )
            missing_x = x_prime[missing_index]

            def teacher_adapter(z_t: Tensor, cond: Tensor, t: Tensor) -> Tensor:
                return teacher_noise_fn(z_t, t, cond, suffix_horizon)

            with torch.no_grad():
                generated = self.diffusion.p_sample_loop(
                    score_net=teacher_adapter,
                    x=missing_x,
                    h=suffix_horizon,
                    batch_size=len(missing),
                    device=x_prime.device,
                )
            result[missing_index] = generated
            if replay_buffer is not None:
                for local_index, batch_index in enumerate(missing):
                    replay_buffer.add(x_prime[batch_index], generated[local_index])
        return result

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, token_dim={self.token_dim}, T={self.schedule.T}"
