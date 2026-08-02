"""
Noise schedules for Bellman Path Diffusion (BPD).

Discrete-time schedule: DDPMSchedule (re-exported from bpd.models.diffusion).
Continuous-time schedule: VPSchedule (VP-SDE, Song et al. 2021).

Forward diffusion boundary conditions (Eq. 19-20):
    alpha_0 = 1,  sigma_0 = 0   (clean signal at t=0)
    alpha_T = 0,  sigma_T = 1   (pure noise at t=T)

References:
    Ho et al. 2020 - "Denoising Diffusion Probabilistic Models" (DDPM)
    Song et al. 2021 - "Score-Based Generative Modeling through SDEs"
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

# DDPMSchedule lives in diffusion.py (canonical implementation used by BlockwiseDiffusion).
# Re-export it here for convenience and backward compatibility.
from bpd.models.diffusion import DDPMSchedule  # noqa: F401


# ---------------------------------------------------------------------------
# Continuous-time VP-SDE schedule
# ---------------------------------------------------------------------------


class VPSchedule:
    """Continuous-time Variance-Preserving (VP) SDE noise schedule.

    Implements the VP-SDE from Song et al. 2021 ("Score-Based Generative
    Modeling through Stochastic Differential Equations"), the continuous-time
    counterpart of the discrete DDPM schedule.

    SDE forward process:
        dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW

    Affine drift:
        beta(t) = beta_min + (t/T) * (beta_max - beta_min)

    Marginal distribution (Eq. 19-20):
        q_{t|0}(z | u) = N(z; alpha_t * phi(u), sigma_t^2 * I)

    with:
        alpha_t = exp(-0.5 * int_0^t beta(s) ds)
        sigma_t = sqrt(1 - alpha_t^2)

    Boundary conditions:
        alpha_0 = 1,  sigma_0 = 0   (exact at t=0)
        alpha_T -> 0, sigma_T -> 1  (approached as beta_max increases)

    For use with the continuous-time reverse SDE (Eq. 36, main paper):
        dx = [-0.5 * beta(t) * x - beta(t) * grad_x log q_t(x)] dt
             + sqrt(beta(t)) dW_bar

    Args:
        beta_min: Minimum noise level at t=0.
        beta_max: Maximum noise level at t=T.
        T:        Terminal diffusion time (default 1.0).
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        T: float = 1.0,
    ) -> None:
        if beta_min <= 0:
            raise ValueError(f"beta_min must be positive; got {beta_min}")
        if beta_max <= beta_min:
            raise ValueError(
                f"beta_max ({beta_max}) must exceed beta_min ({beta_min})"
            )
        if T <= 0:
            raise ValueError(f"T must be positive; got {T}")

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    # ------------------------------------------------------------------
    # Schedule coefficients
    # ------------------------------------------------------------------

    def beta(self, t: float | torch.Tensor) -> float | torch.Tensor:
        """Affine noise coefficient beta(t) = beta_min + t/T*(beta_max - beta_min).

        Args:
            t: Continuous time in [0, T].

        Returns:
            beta(t) of the same type as *t*.
        """
        return self.beta_min + (t / self.T) * (self.beta_max - self.beta_min)

    def _integral_beta(self, t: float | torch.Tensor) -> float | torch.Tensor:
        """Closed-form integral: int_0^t beta(s) ds."""
        return self.beta_min * t + (t ** 2) / (2.0 * self.T) * (
            self.beta_max - self.beta_min
        )

    def alpha_t(self, t: float | torch.Tensor) -> float | torch.Tensor:
        """Signal scaling: alpha_t = exp(-0.5 * int_0^t beta(s) ds).

        Args:
            t: Continuous time in [0, T].

        Returns:
            alpha_t of the same type as *t*.
        """
        integral = self._integral_beta(t)
        if isinstance(t, torch.Tensor):
            return torch.exp(-0.5 * integral)
        return math.exp(-0.5 * integral)

    def sigma_t(self, t: float | torch.Tensor) -> float | torch.Tensor:
        """Noise std: sigma_t = sqrt(1 - alpha_t^2).

        Args:
            t: Continuous time in [0, T].

        Returns:
            sigma_t of the same type as *t*.
        """
        alpha = self.alpha_t(t)
        snr_sq = alpha ** 2
        if isinstance(t, torch.Tensor):
            return torch.sqrt(torch.clamp(1.0 - snr_sq, min=0.0))
        return math.sqrt(max(1.0 - snr_sq, 0.0))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward sample z_t ~ q_{t|0}(z | x0).

            z_t = alpha_t * x0 + sigma_t * eps,   eps ~ N(0, I)

        Args:
            x0:    Clean tensor, shape (B, ...).
            t:     Continuous timesteps in [0, T], shape (B,).
            noise: Optional pre-sampled noise, same shape as x0.

        Returns:
            Tuple (z_t, noise).
        """
        if noise is None:
            noise = torch.randn_like(x0)

        alpha = self.alpha_t(t)
        sigma = self.sigma_t(t)

        while alpha.ndim < x0.ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)

        return alpha * x0 + sigma * noise, noise

    def score_scaling(self, t: float | torch.Tensor) -> float | torch.Tensor:
        """Return sigma_t, the denominator for score = -eps / sigma_t.

        Args:
            t: Continuous time in [0, T].

        Returns:
            sigma_t of the same type as *t*.
        """
        return self.sigma_t(t)

    def __repr__(self) -> str:
        return (
            f"VPSchedule(beta_min={self.beta_min}, "
            f"beta_max={self.beta_max}, T={self.T})"
        )
