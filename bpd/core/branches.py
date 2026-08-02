"""Bellman stop/continuation targets for the discrete DDPM parameterization.

The paper defines score targets in Equations (41)--(47).  Appendix C gives the
equivalent epsilon-prediction form used here:

* stop: predict the sampled Gaussian noise for every block;
* continue: predict the sampled head noise followed by the frozen teacher's
  suffix noise prediction at the same diffusion index.

The continuation probability ``gamma`` is deliberately absent from these
component targets.  It enters once, through sampling the Bellman branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from bpd.models.diffusion import BlockwiseDiffusion

TeacherNoiseFn = Callable[[Tensor, Tensor, Tensor, int], Tensor]


@dataclass(frozen=True)
class BranchTarget:
    """A noisy path and its epsilon-prediction regression target."""

    z_t: Tensor
    target_noise: Tensor
    clean_path: Tensor


class StopBranch:
    """Construct the stopping branch from Equation (41) / Appendix C."""

    def __init__(self, diffusion: BlockwiseDiffusion) -> None:
        self.diffusion = diffusion

    def build(
        self,
        y: Tensor,
        t: Tensor,
        h: int,
        noise: Optional[Tensor] = None,
    ) -> BranchTarget:
        if y.ndim != 2:
            raise ValueError(f"y must have shape (B, d), got {tuple(y.shape)}")
        if h < 1:
            raise ValueError(f"h must be positive, got {h}")
        batch_size, token_dim = y.shape
        if token_dim != self.diffusion.token_dim:
            raise ValueError(
                f"token dimension {token_dim} != diffusion.token_dim "
                f"{self.diffusion.token_dim}"
            )
        if t.shape != (batch_size,):
            raise ValueError(f"t must have shape ({batch_size},), got {tuple(t.shape)}")

        clean_path = y.new_zeros(batch_size, h, token_dim)
        clean_path[:, 0] = y
        if noise is None:
            noise = torch.randn_like(clean_path)
        elif noise.shape != clean_path.shape:
            raise ValueError(
                "noise must have shape "
                f"{tuple(clean_path.shape)}, got {tuple(noise.shape)}"
            )
        z_t = self.diffusion.q_sample_blockwise(clean_path, t, noise=noise)
        return BranchTarget(z_t=z_t, target_noise=noise, clean_path=clean_path)


class ContinueBranch:
    """Construct the continuation branch from Equations (44)--(47).

    ``clean_suffix`` must be sampled from the frozen horizon-``h-1`` teacher
    conditioned on exactly the supplied ``x_prime``.  Forward corruption then
    produces a suffix sample at ``t``; the frozen teacher is evaluated at that
    same noisy suffix and same ``t``.
    """

    def __init__(self, diffusion: BlockwiseDiffusion) -> None:
        self.diffusion = diffusion

    def build(
        self,
        y: Tensor,
        x_prime: Tensor,
        t: Tensor,
        h: int,
        clean_suffix: Tensor,
        teacher_noise_fn: TeacherNoiseFn,
        noise: Optional[Tensor] = None,
    ) -> BranchTarget:
        if h < 2:
            raise ValueError(f"continuation requires h >= 2, got {h}")
        if y.ndim != 2:
            raise ValueError(f"y must have shape (B, d), got {tuple(y.shape)}")
        batch_size, token_dim = y.shape
        expected_suffix = (batch_size, h - 1, token_dim)
        if clean_suffix.shape != expected_suffix:
            raise ValueError(
                f"clean_suffix must have shape {expected_suffix}, "
                f"got {tuple(clean_suffix.shape)}"
            )
        if x_prime.ndim != 2 or x_prime.shape[0] != batch_size:
            raise ValueError(
                f"x_prime must be a matrix with batch size {batch_size}, "
                f"got {tuple(x_prime.shape)}"
            )
        if t.shape != (batch_size,):
            raise ValueError(f"t must have shape ({batch_size},), got {tuple(t.shape)}")

        clean_path = torch.cat((y.unsqueeze(1), clean_suffix), dim=1)
        if noise is None:
            noise = torch.randn_like(clean_path)
        elif noise.shape != clean_path.shape:
            raise ValueError(
                "noise must have shape "
                f"{tuple(clean_path.shape)}, got {tuple(noise.shape)}"
            )
        z_t = self.diffusion.q_sample_blockwise(clean_path, t, noise=noise)

        with torch.no_grad():
            suffix_target = teacher_noise_fn(z_t[:, 1:], t, x_prime, h - 1)
        if suffix_target.shape != expected_suffix:
            raise ValueError(
                f"teacher returned {tuple(suffix_target.shape)}, "
                f"expected {expected_suffix}"
            )

        # Appendix C: no gamma multiplier inside the component target.
        target_noise = torch.cat((noise[:, :1], suffix_target.detach()), dim=1)
        return BranchTarget(
            z_t=z_t,
            target_noise=target_noise,
            clean_path=clean_path,
        )
