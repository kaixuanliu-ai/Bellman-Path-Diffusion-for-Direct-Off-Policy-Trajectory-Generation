"""Stagewise Bellman Path Diffusion training (Algorithm 1).

There is one authoritative loss path: :class:`BellmanDiffusionLoss`.  This
avoids the common but incorrect replacement of the continuation suffix target
with an analytic DSM target for a sampled clean pseudo-trajectory.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None

from bpd.core.branches import TeacherNoiseFn
from bpd.core.objectives import BellmanDiffusionLoss
from bpd.data.dataset import TransitionDataset
from bpd.data.replay import SuffixReplayBuffer
from bpd.models.diffusion import BlockwiseDiffusion
from bpd.training.ema import EMA
from bpd.utils.arrays import DataBatch
from bpd.utils.perf import amp_enabled

logger = logging.getLogger(__name__)


class BellmanPathDiffusionTrainer:
    """Fit horizons ``1,...,H`` with frozen shorter-horizon teachers."""

    def __init__(
        self,
        score_net: nn.Module,
        diffusion: BlockwiseDiffusion,
        objective: BellmanDiffusionLoss,
        optimizer: Optimizer,
        ema: EMA,
        config: Optional[Dict] = None,
    ) -> None:
        if objective.diffusion is not diffusion:
            raise ValueError("objective and trainer must share one diffusion instance")
        self.score_net = score_net
        self.diffusion = diffusion
        self.objective = objective
        self.optimizer = optimizer
        self.ema = ema
        self.config = dict(config or {})

        self.gamma = float(self.config.get("gamma", objective.gamma))
        if abs(self.gamma - objective.gamma) > 1e-12:
            raise ValueError("trainer gamma differs from objective gamma")
        self.H = int(self.config.get("H", 4))
        self.steps_per_horizon = int(self.config.get("steps_per_horizon", 100_000))
        self.batch_size = int(self.config.get("batch_size", 256))
        self.log_freq = int(self.config.get("log_freq", 1_000))
        self.save_freq = int(self.config.get("save_freq", 10_000))
        self.replay_buffer_size = int(self.config.get("replay_buffer_size", 50_000))
        self.replay_refresh_size = int(
            self.config.get(
                "replay_refresh_size", min(self.replay_buffer_size, self.batch_size)
            )
        )
        self.replay_refresh_freq = int(self.config.get("replay_refresh_freq", 5_000))
        self.grad_clip_norm = float(self.config.get("grad_clip_norm", 1.0))
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        # GPU throughput knobs (no-ops on CPU).
        self.amp = bool(self.config.get("amp", True))
        self.data_on_device = bool(self.config.get("data_on_device", True))
        # Replay amortization (Proposition 2): fill the buffer with clean
        # teacher suffixes for a pool of transitions once per refresh, then draw
        # training minibatches from that pool so the continuation branch reuses
        # cached suffixes (corrupt-at-t) instead of regenerating them every
        # step.  Without this, continuous x' always misses the exact-key buffer
        # and each step pays a full T-step teacher rollout.  Set False for the
        # strict per-step fresh-sampling path (population-equivalent but slow).
        self.amortize_replay = bool(self.config.get("amortize_replay", True))

        if self.H < 1 or self.steps_per_horizon < 1 or self.batch_size < 1:
            raise ValueError("H, steps_per_horizon, and batch_size must be positive")
        self._teachers: Dict[int, nn.Module] = {}
        self._replay_buffers: Dict[int, SuffixReplayBuffer] = {}
        self._global_step = 0
        self._scaler: Optional[torch.amp.GradScaler] = None

    def train_all_stages(
        self,
        dataset: TransitionDataset,
        eval_policy_fn: Callable[[Tensor], Tensor],
        device: torch.device,
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        for h in range(1, self.H + 1):
            metrics = self.train_stage(h, dataset, eval_policy_fn, device, writer)
            logger.info(
                "completed horizon %d/%d: final_loss=%.6f mean_loss=%.6f",
                h,
                self.H,
                metrics["final_loss"],
                metrics["mean_loss"],
            )

    def train_stage(
        self,
        h: int,
        dataset: TransitionDataset,
        eval_policy_fn: Callable[[Tensor], Tensor],
        device: torch.device,
        writer: Optional[SummaryWriter] = None,
    ) -> Dict[str, float]:
        if not 1 <= h <= self.H:
            raise ValueError(f"h must lie in [1, {self.H}], got {h}")
        device = torch.device(device)
        self.score_net.to(device).train()
        self.diffusion.to(device)
        # The EMA shadow (and every frozen teacher) must live on the compute
        # device: the shadow is the continuation-branch teacher and is queried
        # on device tensors during replay refresh and training.
        self.ema.get_model().to(device)
        for teacher in self._teachers.values():
            teacher.to(device)
        # Keep the transition dataset resident on the compute device so batch
        # sampling is an on-device gather (no per-step host->device copies).
        if self.data_on_device and dataset.states.device != device:
            dataset.to(device)
        # Mixed precision (CUDA only); GradScaler is created once and reused.
        self._amp_on = amp_enabled(device, self.amp)
        if self._scaler is None:
            self._scaler = torch.amp.GradScaler(
                device.type, enabled=self._amp_on
            )
        teacher_fn = self._teacher_noise_fn(h)

        replay: Optional[SuffixReplayBuffer] = None
        pool: Optional[tuple[Tensor, Tensor]] = None
        if h > 1:
            replay = SuffixReplayBuffer(
                max_size=self.replay_buffer_size,
                suffix_horizon=h - 1,
                token_dim=self.diffusion.token_dim,
            )
            self._replay_buffers[h] = replay
            pool = self._refresh_replay_buffer(
                replay,
                h,
                dataset,
                eval_policy_fn,
                teacher_fn,
                device,
                self.replay_refresh_size,
            )

        # When amortizing, training draws from the buffered pool so the
        # continuation branch reuses cached suffixes instead of regenerating.
        sample_pool = pool if self.amortize_replay else None

        running_loss = 0.0
        stage_loss = 0.0
        final_loss = float("nan")
        for local_step in range(1, self.steps_per_horizon + 1):
            self._global_step += 1
            batch, next_action, cached_suffix = self._sample_batch(
                dataset, eval_policy_fn, device, pool=sample_pool
            )
            with torch.autocast(device_type=device.type, enabled=self._amp_on):
                loss = self.objective(
                    self.score_net,
                    teacher_fn,
                    batch,
                    h,
                    next_action=next_action,
                    replay_buffer=replay,
                    cached_suffix=cached_suffix,
                )

            self.optimizer.zero_grad(set_to_none=True)
            self._scaler.scale(loss).backward()
            if self.grad_clip_norm > 0:
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.score_net.parameters(), self.grad_clip_norm
                )
            self._scaler.step(self.optimizer)
            self._scaler.update()
            self.ema.update(self.score_net, step=self._global_step)

            final_loss = float(loss.detach().item())
            stage_loss += final_loss
            running_loss += final_loss
            if local_step % self.log_freq == 0:
                mean = running_loss / self.log_freq
                logger.info("h=%d step=%d loss=%.6f", h, self._global_step, mean)
                if writer is not None:
                    writer.add_scalar(f"train/loss_h{h}", mean, self._global_step)
                running_loss = 0.0

            if self.save_freq > 0 and local_step % self.save_freq == 0:
                self.save(self._checkpoint_path(h, self._global_step), h)

            if (
                replay is not None
                and self.replay_refresh_freq > 0
                and local_step % self.replay_refresh_freq == 0
            ):
                pool = self._refresh_replay_buffer(
                    replay,
                    h,
                    dataset,
                    eval_policy_fn,
                    teacher_fn,
                    device,
                    self.replay_refresh_size,
                )
                sample_pool = pool if self.amortize_replay else None

        self._freeze_stage(h)
        return {
            "final_loss": final_loss,
            "mean_loss": stage_loss / self.steps_per_horizon,
            "steps": float(self.steps_per_horizon),
        }

    def _sample_batch(
        self,
        dataset: TransitionDataset,
        eval_policy_fn: Callable[[Tensor], Tensor],
        device: torch.device,
        pool: Optional[tuple[Tensor, Tensor, Tensor]] = None,
    ) -> tuple[DataBatch, Tensor, Optional[Tensor]]:
        if pool is not None:
            # Amortized path: draw from the replay pool and return the aligned
            # cached suffixes so the objective skips the buffer lookup entirely
            # (Proposition 2).
            pool_index, pool_next_action, pool_suffix = pool
            sel = torch.randint(
                0, pool_index.shape[0], (self.batch_size,), device=pool_index.device
            )
            index = pool_index[sel]
            next_action = pool_next_action[sel].to(device)
            cached_suffix = pool_suffix[sel].to(device)
            batch = DataBatch(
                state=dataset.states[index].to(device, non_blocking=True),
                action=dataset.actions[index].to(device, non_blocking=True),
                reward=dataset.rewards[index].to(device, non_blocking=True),
                next_state=dataset.next_states[index].to(device, non_blocking=True),
                done=dataset.dones[index].to(device, non_blocking=True),
            )
            return batch, next_action, cached_suffix

        index = torch.randint(
            0, len(dataset), (self.batch_size,), device=dataset.states.device
        )
        batch = DataBatch(
            state=dataset.states[index].to(device, non_blocking=True),
            action=dataset.actions[index].to(device, non_blocking=True),
            reward=dataset.rewards[index].to(device, non_blocking=True),
            next_state=dataset.next_states[index].to(device, non_blocking=True),
            done=dataset.dones[index].to(device, non_blocking=True),
        )
        with torch.no_grad():
            next_action = eval_policy_fn(batch.next_state)
        expected = (self.batch_size, dataset.act_dim)
        if next_action.shape != expected:
            raise ValueError(
                f"evaluation policy returned {tuple(next_action.shape)}, "
                f"expected {expected}"
            )
        next_action = next_action.to(device=device, dtype=batch.action.dtype)
        next_action = torch.where(
            batch.done.bool().unsqueeze(-1),
            torch.zeros_like(next_action),
            next_action,
        )
        return batch, next_action, None

    @torch.no_grad()
    def _refresh_replay_buffer(
        self,
        replay: SuffixReplayBuffer,
        h: int,
        dataset: TransitionDataset,
        eval_policy_fn: Callable[[Tensor], Tensor],
        teacher_fn: Optional[TeacherNoiseFn],
        device: torch.device,
        n_samples: int,
    ) -> Optional[tuple[Tensor, Tensor, Tensor]]:
        """Generate clean teacher suffixes for a pool of transitions and cache
        them, keyed by exactly the (s', a') used to condition the generator.

        Returns the pool as ``(indices, next_actions, suffixes)`` so that
        training can draw minibatches from the same transitions and hand the
        aligned clean suffixes straight to the objective, amortizing the teacher
        rollout over many steps.
        """
        if h <= 1 or n_samples <= 0:
            return None
        if teacher_fn is None:
            raise RuntimeError(f"missing frozen teacher for horizon {h - 1}")
        index = torch.randint(
            0, len(dataset), (n_samples,), device=dataset.next_states.device
        )
        next_state = dataset.next_states[index].to(device)
        done = dataset.dones[index].to(device).bool()
        next_action = eval_policy_fn(next_state).to(
            device=device, dtype=next_state.dtype
        )
        next_action = torch.where(
            done.unsqueeze(-1), torch.zeros_like(next_action), next_action
        )
        x_prime = torch.cat((next_state, next_action), dim=-1)
        suffix_horizon = h - 1

        def adapter(z_t: Tensor, condition: Tensor, t: Tensor) -> Tensor:
            return teacher_fn(z_t, t, condition, suffix_horizon)

        suffix = self.diffusion.p_sample_loop(
            score_net=adapter,
            x=x_prime,
            h=suffix_horizon,
            batch_size=n_samples,
            device=device,
        )
        # Move the whole batch to host in ONE transfer before inserting: the
        # buffer stores CPU copies, so a per-item add() would issue n_samples
        # separate device->host copies (a GPU stall each) during every refresh.
        x_prime_cpu = x_prime.detach().to("cpu", non_blocking=False)
        suffix_cpu = suffix.detach().to("cpu", non_blocking=False)
        for sample_index in range(n_samples):
            # The key and generator condition are exactly the same (s', a').
            replay.add(x_prime_cpu[sample_index], suffix_cpu[sample_index])
        # Pool for amortized training reuse: the transitions whose clean
        # suffixes are now cached.  The suffix tensor is returned so training
        # can hand it directly to the objective (fast path, no per-item buffer
        # lookup); next_action is kept so stochastic policies still reproduce
        # the exact buffered x'.
        return index.to(device), next_action, suffix

    def _teacher_noise_fn(self, h: int) -> Optional[TeacherNoiseFn]:
        if h == 1:
            return None
        teacher_horizon = h - 1
        if teacher_horizon not in self._teachers:
            raise RuntimeError(
                f"horizon {teacher_horizon} teacher is unavailable; "
                "train stages sequentially"
            )
        teacher = self._teachers[teacher_horizon]

        def call(z_t: Tensor, t: Tensor, x: Tensor, requested_horizon: int) -> Tensor:
            if requested_horizon != teacher_horizon:
                raise ValueError(
                    f"teacher h={teacher_horizon} was queried at h={requested_horizon}"
                )
            return teacher(z_t, t, x, teacher_horizon)

        return call

    def _freeze_stage(self, h: int) -> None:
        device = next(self.score_net.parameters()).device
        teacher = copy.deepcopy(self.ema.get_model()).to(device)
        teacher.eval().requires_grad_(False)
        self._teachers[h] = teacher

    def _checkpoint_path(self, h: int, step: int) -> Path:
        return self.checkpoint_dir / f"bpd_h{h:02d}_step{step:07d}.pt"

    def save(self, path: str | Path, h: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "score_net": self.score_net.state_dict(),
                "ema": self.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "teachers": {
                    stage: net.state_dict() for stage, net in self._teachers.items()
                },
                "global_step": self._global_step,
                "h": h,
                "config": self.config,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location="cpu")
        self.score_net.load_state_dict(checkpoint["score_net"])
        self.ema.load_state_dict(checkpoint["ema"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._global_step = int(checkpoint.get("global_step", 0))
        self._teachers.clear()
        for stage, state in checkpoint.get("teachers", {}).items():
            teacher = copy.deepcopy(self.score_net)
            teacher.load_state_dict(state)
            teacher.eval().requires_grad_(False)
            self._teachers[int(stage)] = teacher


def diffusion_token_dim(diffusion: BlockwiseDiffusion) -> int:
    """Compatibility helper retained for external callers."""
    return diffusion.token_dim
