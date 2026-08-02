"""
Blockwise forward diffusion and reverse SDE solver for Bellman Path Diffusion (BPD).

Implements the diffusion machinery from:
  "Bellman Path Diffusion for Direct Off-Policy Trajectory Generation"

Key equations implemented here:

  Blockwise corruption kernel (Eq. 24):
      q_{t|0}^{(h)}(dz_{0:h-1} | w) = ∏_{j=0}^{h-1} q_{t|0}(dz_j | w_j)

  Noisy path law (Eq. 25):
      m_{h,t}^π = (Diff_{h,t})_# M_h^π

  Reverse SDE (Eq. 36):
      dZ_t = [f(t) Z_t  -  g(t)² ∇_z log m_{h,t}^π(Z_t | x)] dt  +  g(t) dW̄_t

  Score target for analytic q_{t|0}:
      ∇_z log q_{t|0}(z | w_j) = -1/σ_t² * (z - α_t * w_j)   (Eq. 43 / Tweedie)

Discrete-time DDPM parameterisation follows:
  Ho et al. 2020 – "Denoising Diffusion Probabilistic Models"
  Janner et al. 2022 – "Planning with Diffusion"

Schedule convention (variance-preserving, VP-SDE / DDPM):
    q_{t|0}(z | w) = N(α_t * w,  σ_t² I)
    α_t = sqrt(ᾱ_t)
    σ_t = sqrt(1 - ᾱ_t)
    ᾱ_t = ∏_{s=1}^{t} (1 - β_s)

Shapes used throughout:
    B          – batch size
    h          – path horizon (number of tokens per path)
    token_dim  – d_tok = 1 + obs_dim + act_dim
    T          – total diffusion timesteps (e.g. 1000)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------


@dataclass
class DDPMSchedule:
    """Discrete DDPM noise schedule (linear or cosine).

    Stores precomputed cumulative products ᾱ_t and derived quantities
    for timesteps t = 0, 1, ..., T.

    Attributes:
        T:          Total number of diffusion steps.
        betas:      β_t for t = 1 ... T, shape (T,).
        alphas:     1 - β_t, shape (T,).
        alphas_bar: ᾱ_t = ∏_{s=1}^{t} α_s, shape (T+1,) with ᾱ_0 = 1.
        sqrt_alphas_bar:      α_t = sqrt(ᾱ_t), shape (T+1,).
        sqrt_one_minus_alphas_bar:  σ_t = sqrt(1 - ᾱ_t), shape (T+1,).
        sqrt_recip_alphas_bar:      1 / α_t,  shape (T+1,).
        posterior_variance:   β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t),
                              shape (T,) for t = 1 ... T.
        posterior_mean_coef1: coefficient of z_0 in posterior mean,
                              shape (T,).
        posterior_mean_coef2: coefficient of z_t in posterior mean,
                              shape (T,).
    """

    T: int
    betas: Tensor
    alphas: Tensor
    alphas_bar: Tensor
    sqrt_alphas_bar: Tensor
    sqrt_one_minus_alphas_bar: Tensor
    sqrt_recip_alphas_bar: Tensor
    posterior_variance: Tensor
    posterior_mean_coef1: Tensor
    posterior_mean_coef2: Tensor

    @classmethod
    def make_linear(
        cls,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> "DDPMSchedule":
        """Linear schedule: β_t linearly interpolated from beta_start to beta_end.

        Args:
            T:          Number of diffusion steps.
            beta_start: β at t = 1.
            beta_end:   β at t = T.

        Returns:
            Fully initialised DDPMSchedule.
        """
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        return cls._from_betas(T, betas)

    @classmethod
    def make_cosine(
        cls,
        T: int = 1000,
        s: float = 0.008,
    ) -> "DDPMSchedule":
        """Cosine schedule (Nichol & Dhariwal 2021).

        ᾱ_t = f(t) / f(0),  f(t) = cos²( (t/T + s) / (1 + s) * π/2 )

        Args:
            T: Number of diffusion steps.
            s: Offset hyperparameter (default 0.008).

        Returns:
            Fully initialised DDPMSchedule.
        """
        steps = T + 1  # t = 0, 1, ..., T
        t = torch.arange(steps, dtype=torch.float64)
        f = torch.cos(((t / T) + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alphas_bar_full = f / f[0]

        # β_t = 1 - ᾱ_t / ᾱ_{t-1}, clipped to [0, 0.999]
        betas = 1.0 - (alphas_bar_full[1:] / alphas_bar_full[:-1])
        betas = betas.clamp(0.0, 0.999)
        return cls._from_betas(T, betas)

    @classmethod
    def _from_betas(cls, T: int, betas: Tensor) -> "DDPMSchedule":
        betas = betas.float()
        alphas = 1.0 - betas

        # ᾱ_0 = 1 (no noise at t=0)
        alphas_bar = torch.cat(
            [torch.ones(1, dtype=torch.float32), torch.cumprod(alphas, dim=0)],
            dim=0,
        )  # shape (T+1,)

        sqrt_alphas_bar = torch.sqrt(alphas_bar)
        sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)
        sqrt_recip_alphas_bar = 1.0 / sqrt_alphas_bar.clamp(min=1e-8)

        # DDPM posterior q(z_{t-1} | z_t, z_0):
        #   mean = coef1 * z_0 + coef2 * z_t
        #   var  = β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        #
        # Indices: betas[i] = β_{i+1}  (i = 0 ... T-1)
        #          alphas_bar[t] = ᾱ_t
        alphas_bar_prev = alphas_bar[:-1]  # ᾱ_{t-1} for t = 1 ... T
        posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar[1:]).clamp(min=1e-8)

        posterior_mean_coef1 = (
            torch.sqrt(alphas_bar_prev) * betas / (1.0 - alphas_bar[1:]).clamp(min=1e-8)
        )
        posterior_mean_coef2 = (
            torch.sqrt(alphas) * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar[1:]).clamp(min=1e-8)
        )

        return cls(
            T=T,
            betas=betas,
            alphas=alphas,
            alphas_bar=alphas_bar,
            sqrt_alphas_bar=sqrt_alphas_bar,
            sqrt_one_minus_alphas_bar=sqrt_one_minus_alphas_bar,
            sqrt_recip_alphas_bar=sqrt_recip_alphas_bar,
            posterior_variance=posterior_variance,
            posterior_mean_coef1=posterior_mean_coef1,
            posterior_mean_coef2=posterior_mean_coef2,
        )

    # ------------------------------------------------------------------
    # Convenience accessors (index with Python int t = 1 ... T)
    # ------------------------------------------------------------------

    def alpha_bar(self, t: int) -> float:
        """ᾱ_t as a Python float."""
        return float(self.alphas_bar[t].item())

    def sigma(self, t: int) -> float:
        """σ_t = sqrt(1 - ᾱ_t)."""
        return float(self.sqrt_one_minus_alphas_bar[t].item())

    def alpha(self, t: int) -> float:
        """α_t = sqrt(ᾱ_t)."""
        return float(self.sqrt_alphas_bar[t].item())


# ---------------------------------------------------------------------------
# Score-network protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ScoreNet(Protocol):
    """Protocol for a score / noise-prediction network.

    Following Ho et al. 2020, the network predicts the noise ε that was added
    (epsilon-parameterisation), *not* the score directly.  The analytic
    relationship is:

        score(z, x, t) = -ε_θ(z, x, t) / σ_t

    Args (call signature):
        z_t:  Noisy trajectory tokens, shape (B, h, token_dim).
        x:    Conditioning context (e.g. current state), shape (B, obs_dim).
        t:    Discrete timestep indices, shape (B,), dtype long.

    Returns:
        Predicted noise ε, same shape as z_t: (B, h, token_dim).
    """

    def __call__(self, z_t: Tensor, x: Tensor, t: Tensor) -> Tensor:
        ...


# ---------------------------------------------------------------------------
# BlockwiseDiffusion
# ---------------------------------------------------------------------------


class BlockwiseDiffusion(nn.Module):
    """Blockwise DDPM diffusion for Bellman Path Diffusion.

    Implements the forward corruption kernel (Eq. 24) and the full DDPM
    reverse chain needed for trajectory sampling.

    Each path w ∈ W_h is a sequence of h token vectors in R^{d_tok}.
    Padding tokens are encoded as zeros (see bpd/core/path.py).

    The blockwise kernel treats each token *independently* under the *same*
    noise level t:

        q_{t|0}^{(h)}(dz_{0:h-1} | w) = ∏_{j=0}^{h-1} q_{t|0}(dz_j | w_j)

    so the joint forward pass is just a single vectorised operation over the
    h-dimension.

    Args:
        schedule:   Pre-built DDPMSchedule (linear or cosine).
        token_dim:  Dimensionality of each token d_tok.

    Note on score-network convention:
        ``score_net`` is expected to predict *noise* ε_θ (epsilon-prediction).
        Internally we convert ε → score via ``score = -ε / σ_t``.
    """

    def __init__(self, schedule: DDPMSchedule, token_dim: int) -> None:
        super().__init__()
        self.schedule = schedule
        self.token_dim = token_dim
        self.T = schedule.T

        # Register all schedule tensors as buffers so they move with .to(device)
        self.register_buffer("alphas_bar", schedule.alphas_bar)
        self.register_buffer("sqrt_alphas_bar", schedule.sqrt_alphas_bar)
        self.register_buffer("sqrt_one_minus_alphas_bar", schedule.sqrt_one_minus_alphas_bar)
        self.register_buffer("betas", schedule.betas)
        self.register_buffer("alphas", schedule.alphas)
        self.register_buffer("posterior_variance", schedule.posterior_variance)
        self.register_buffer("posterior_mean_coef1", schedule.posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", schedule.posterior_mean_coef2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, a: Tensor, t: Tensor, broadcast_shape: tuple) -> Tensor:
        """Gather schedule values at timesteps t and broadcast to target shape.

        Args:
            a:               1-D schedule tensor of shape (T+1,) or (T,).
            t:               Integer timestep indices, shape (B,).
            broadcast_shape: Target shape for broadcasting, e.g. (B, h, d).

        Returns:
            Tensor of shape ``broadcast_shape`` with values a[t[b]] for each b.
        """
        # t indices may be 1-based (t=1..T) or 0-based depending on caller;
        # the caller is responsible for correct indexing into `a`.
        out = a[t]  # (B,)
        # Reshape to (B, 1, 1, ...) and expand
        while out.dim() < len(broadcast_shape):
            out = out.unsqueeze(-1)
        return out.expand(broadcast_shape)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample_blockwise(
        self,
        w: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the blockwise forward corruption kernel (Eq. 24).

        Samples z_{0:h-1} ~ q_{t|0}^{(h)}(· | w) by applying the same
        noise level t independently to every token block:

            z_j = α_t * w_j + σ_t * ε_j,   ε_j ~ N(0, I_{d_tok})

        Padding tokens (encoded as zeros) are corrupted identically to real
        tokens — the network must learn to denoise them back to zero.

        Args:
            w:     Clean trajectory tokens, shape (B, h, token_dim).
                   Padding positions should already be zero-encoded.
            t:     Discrete timestep indices in {1, ..., T}, shape (B,).
            noise: Optional pre-sampled Gaussian noise of shape (B, h, token_dim).
                   If None, freshly sampled from N(0, I).

        Returns:
            Noisy trajectory z_t of shape (B, h, token_dim).
        """
        B, h, d = w.shape
        assert d == self.token_dim, (
            f"token_dim mismatch: w has dim {d}, expected {self.token_dim}"
        )
        assert t.shape == (B,), f"t must have shape (B,)={( B,)}; got {t.shape}"

        if noise is None:
            noise = torch.randn_like(w)  # (B, h, token_dim)

        shape = (B, h, d)

        # α_t = sqrt(ᾱ_t)
        sqrt_ab = self._extract(self.sqrt_alphas_bar, t, shape)
        # σ_t = sqrt(1 - ᾱ_t)
        sqrt_one_minus_ab = self._extract(self.sqrt_one_minus_alphas_bar, t, shape)

        # z_t = α_t * w + σ_t * ε
        z_t = sqrt_ab * w + sqrt_one_minus_ab * noise
        return z_t

    # ------------------------------------------------------------------
    # Score targets
    # ------------------------------------------------------------------

    def blockwise_score_target(
        self,
        z: Tensor,
        w: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Analytic score of the blockwise forward kernel q_{t|0}^{(h)}(z | w).

        Since each block is an independent Gaussian:

            q_{t|0}(z_j | w_j) = N(z_j; α_t * w_j, σ_t² I)

        the score factorises as:

            ∇_{z_j} log q_{t|0}(z_j | w_j) = -1/σ_t² * (z_j - α_t * w_j)

        This closed-form score is used as:
        - The score target for the *stop* branch (Eq. 43, BPD paper).
        - The score contribution from the *current token* in the continue branch.

        Padding tokens (w_j = 0) contribute ∇ log q = -z_j / σ_t², pulling the
        reverse chain back towards zero, which is correct by construction.

        Args:
            z:  Noisy trajectory tokens, shape (B, h, token_dim).
            w:  Clean reference tokens, shape (B, h, token_dim).
            t:  Discrete timestep indices in {1, ..., T}, shape (B,).

        Returns:
            Score tensor of shape (B, h, token_dim).
        """
        B, h, d = z.shape
        assert z.shape == w.shape, (
            f"z and w must have the same shape; got {z.shape} vs {w.shape}"
        )
        assert t.shape == (B,), f"t must have shape (B,)={(B,)}; got {t.shape}"

        shape = (B, h, d)

        # σ_t = sqrt(1 - ᾱ_t)
        sigma_t = self._extract(self.sqrt_one_minus_alphas_bar, t, shape)  # (B,h,d)
        sigma_t_sq = sigma_t ** 2

        # α_t = sqrt(ᾱ_t)
        alpha_t = self._extract(self.sqrt_alphas_bar, t, shape)  # (B,h,d)

        # score = -1/σ_t² * (z - α_t * w)
        score = -(z - alpha_t * w) / sigma_t_sq.clamp(min=1e-8)
        return score

    # ------------------------------------------------------------------
    # DDPM posterior (used in reverse step)
    # ------------------------------------------------------------------

    def _predict_z0_from_noise(
        self,
        z_t: Tensor,
        t: Tensor,
        noise_pred: Tensor,
    ) -> Tensor:
        """Recover z_0 estimate from predicted noise via Tweedie formula.

        z_0 = (z_t - σ_t * ε_θ) / α_t

        Args:
            z_t:        Noisy samples, shape (B, h, d).
            t:          Timesteps, shape (B,).
            noise_pred: Predicted noise from score_net, shape (B, h, d).

        Returns:
            Estimated z_0, shape (B, h, d).
        """
        shape = z_t.shape
        sqrt_ab = self._extract(self.sqrt_alphas_bar, t, shape)
        sqrt_one_minus_ab = self._extract(self.sqrt_one_minus_alphas_bar, t, shape)

        # z_0 = (z_t - σ_t * ε) / α_t
        z0_hat = (z_t - sqrt_one_minus_ab * noise_pred) / sqrt_ab.clamp(min=1e-8)
        return z0_hat

    def _q_posterior_mean(
        self,
        z_t: Tensor,
        z0_hat: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute the DDPM posterior mean μ̃_t(z_t, z_0).

        μ̃_t = coef1 * z_0 + coef2 * z_t

        Args:
            z_t:    Noisy sample at time t, shape (B, h, d).
            z0_hat: Estimated clean sample, shape (B, h, d).
            t:      Timesteps (1-based, t = 1 ... T), shape (B,).

        Returns:
            Posterior mean, shape (B, h, d).
        """
        shape = z_t.shape
        # posterior_mean_coef1/2 are indexed for t = 1 ... T,
        # so we use t-1 as 0-based index into these (T,) tensors.
        t_idx = (t - 1).clamp(min=0)
        coef1 = self._extract(self.posterior_mean_coef1, t_idx, shape)
        coef2 = self._extract(self.posterior_mean_coef2, t_idx, shape)
        mean = coef1 * z0_hat + coef2 * z_t
        return mean

    # ------------------------------------------------------------------
    # Single reverse step
    # ------------------------------------------------------------------

    def p_sample_step(
        self,
        score_net_output: Tensor,
        z_t: Tensor,
        t: int,
        x: Tensor,
    ) -> Tensor:
        """One DDPM reverse step: z_t → z_{t-1}.

        Implements the discrete reverse transition:

            p_θ(z_{t-1} | z_t, x) = N(z_{t-1}; μ̃_θ(z_t, x, t), β̃_t I)

        where:
            μ̃_θ(z_t, x, t) = coef1 * ẑ_0 + coef2 * z_t
            ẑ_0 = (z_t - σ_t * ε_θ) / α_t   (Tweedie / epsilon-prediction)

        At t = 1, no noise is added (deterministic final step).

        Args:
            score_net_output: Predicted noise ε_θ from the score/noise network,
                              shape (B, h, token_dim).
            z_t:              Current noisy trajectory, shape (B, h, token_dim).
            t:                Current discrete timestep (Python int, 1 ≤ t ≤ T).
            x:                Conditioning context, shape (B, obs_dim).
                              Not used in this base method but kept for API
                              compatibility with conditional models.

        Returns:
            Denoised sample z_{t-1}, shape (B, h, token_dim).
        """
        B, h, d = z_t.shape
        device = z_t.device
        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)

        # Estimate z_0 from predicted noise
        z0_hat = self._predict_z0_from_noise(z_t, t_tensor, score_net_output)
        # Clip estimated z_0 for stability (standard DDPM practice)
        z0_hat = z0_hat.clamp(-1.0, 1.0)

        # Compute posterior mean μ̃_θ
        mean = self._q_posterior_mean(z_t, z0_hat, t_tensor)

        if t == 1:
            # Deterministic final step — no noise added
            return mean

        # Sample z_{t-1} = μ̃_θ + sqrt(β̃_t) * ε,  ε ~ N(0, I)
        # posterior_variance[i] = β̃_{i+1}, so index t-1 gives β̃_t
        var_t = float(self.posterior_variance[t - 1].item())
        noise = torch.randn_like(z_t)
        z_prev = mean + math.sqrt(max(var_t, 0.0)) * noise
        return z_prev

    # ------------------------------------------------------------------
    # Full reverse chain (trajectory generation)
    # ------------------------------------------------------------------

    def p_sample_loop(
        self,
        score_net: ScoreNet,
        x: Tensor,
        h: int,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Full DDPM reverse chain: N(0, I_{h*d}) → clean trajectory.

        Iterates the reverse Markov chain from t = T down to t = 1,
        calling the score network at each step, to produce samples
        approximately distributed as M_h^π(· | x).

        Args:
            score_net:  A callable satisfying the ScoreNet protocol.
                        Takes (z_t, x, t) and returns predicted noise of
                        shape (B, h, token_dim).
            x:          Conditioning context (e.g. current state observation),
                        shape (batch_size, obs_dim).
            h:          Path horizon (number of token blocks).
            batch_size: Number of trajectories to generate in parallel.
            device:     Target compute device.

        Returns:
            Clean trajectory samples of shape (batch_size, h, token_dim).
        """
        # Start from pure Gaussian noise
        z = torch.randn(batch_size, h, self.token_dim, device=device)

        for t in reversed(range(1, self.T + 1)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)

            with torch.no_grad():
                noise_pred = score_net(z, x, t_tensor)

            z = self.p_sample_step(noise_pred, z, t, x)

        return z

    # ------------------------------------------------------------------
    # Partial reverse integration (suffix sampling)
    # ------------------------------------------------------------------

    def p_sample_partial(
        self,
        score_net: ScoreNet,
        x: Tensor,
        h: int,
        t_stop: int,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Partial reverse integration: N(0, I) → noise level t_stop.

        Runs the reverse chain from t = T down to t = t_stop + 1,
        stopping before the final denoising steps.  The returned sample
        z_{t_stop} retains a controlled amount of noise and can be used
        as an initialisation for suffix / conditional sampling schemes
        (e.g., repaint-style inpainting over future trajectory blocks).

        Args:
            score_net:  Score / noise-prediction network.
            x:          Conditioning context, shape (batch_size, obs_dim).
            h:          Path horizon.
            t_stop:     Timestep at which to stop (inclusive).
                        The returned sample is z_{t_stop}.
                        Must satisfy 1 ≤ t_stop < T.
            batch_size: Number of samples to generate.
            device:     Compute device.

        Returns:
            Noisy trajectory at noise level t_stop,
            shape (batch_size, h, token_dim).
        """
        if not (1 <= t_stop < self.T):
            raise ValueError(
                f"t_stop must satisfy 1 ≤ t_stop < T={self.T}; got {t_stop}"
            )

        z = torch.randn(batch_size, h, self.token_dim, device=device)

        for t in reversed(range(t_stop + 1, self.T + 1)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)

            with torch.no_grad():
                noise_pred = score_net(z, x, t_tensor)

            z = self.p_sample_step(noise_pred, z, t, x)

        return z

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def training_losses(
        self,
        score_net: nn.Module,
        w: Tensor,
        x: Tensor,
        t: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Compute the DDPM training loss (simple ε-prediction objective).

        L_simple = E_{t,ε}[ || ε - ε_θ(z_t, x, t) ||² ]

        This is the objective used in Ho et al. 2020 (Eq. 14).  For BPD it
        serves as the denoising score-matching loss for the blockwise kernel.

        Args:
            score_net: Noise-prediction network ε_θ.
            w:         Clean trajectory tokens, shape (B, h, token_dim).
            x:         Conditioning context, shape (B, obs_dim).
            t:         Optional pre-sampled timesteps, shape (B,).
                       If None, uniformly sampled from {1, ..., T}.
            noise:     Optional pre-sampled Gaussian noise, shape (B, h, d).
                       If None, freshly sampled.

        Returns:
            Dictionary with keys:
                ``"loss"``    – scalar mean squared error.
                ``"noise"``   – the target noise ε used (for logging).
                ``"z_t"``     – the noisy sample used as input to score_net.
                ``"t"``       – the timestep indices used.
        """
        B, h, d = w.shape
        device = w.device

        if t is None:
            t = torch.randint(1, self.T + 1, (B,), device=device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(w)

        z_t = self.q_sample_blockwise(w, t, noise=noise)
        noise_pred = score_net(z_t, x, t)

        loss = ((noise_pred - noise) ** 2).mean()

        return {
            "loss": loss,
            "noise": noise,
            "z_t": z_t,
            "t": t,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BlockwiseDiffusion("
            f"T={self.T}, "
            f"token_dim={self.token_dim})"
        )
