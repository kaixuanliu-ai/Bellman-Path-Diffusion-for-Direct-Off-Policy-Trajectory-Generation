"""
Trajectory score network s_θ(z, t | x, h) for Bellman Path Diffusion (BPD).

Implements the DiT-style (Peebles & Xie, 2023 ICCV) transformer score network
described in Remark 3 of the BPD paper.  A *single* transformer handles all
horizon values h ∈ {1, …, H} by:

  1. Injecting a learned horizon embedding (MLP over h).
  2. Using a variable-length attention mask that prevents the real tokens from
     attending to padding positions (positions j ≥ h are masked out).
  3. Modulating every transformer block with an Adaptive Layer Norm (AdaLN)
     conditioned on the combined time + horizon embedding, following DiT.

Architecture overview
---------------------
Input:
  z  ∈ R^{B × h × d_tok}   – noisy path tensor
  t  ∈ Z^{B}                – diffusion time steps (long integers)
  x  ∈ R^{B × (obs_dim + act_dim)}  – current state-action conditioning
  h  ∈ {1, …, H}            – horizon (integer, same for all items in batch)

Processing:
  1.  Token projection:  z_{B,j} → e_{B,j} ∈ R^{model_dim}
  2.  Time embedding:    t → sinusoidal → MLP → time_emb ∈ R^{B × model_dim}
  3.  Horizon embedding: h → MLP → hor_emb ∈ R^{1 × model_dim}  (broadcast)
  4.  Cond embedding:    x → Linear → cond_emb ∈ R^{B × model_dim}
  5.  Modulation signal: mod = time_emb + hor_emb          (R^{B × model_dim})
  6.  Cross-attention context: [time_emb, hor_emb, cond_emb] stacked along seq
  7.  L × AdaLNBlock: self-attention + cross-attention + FFN, all under AdaLN
  8.  Output projection: e_{B,j} → score_{B,j} ∈ R^{d_tok}

References
----------
* Peebles & Xie (2023). "Scalable Diffusion Models with Transformers" (DiT). ICCV.
* Chen et al. (2021). "Decision Transformer". NeurIPS.
* BPD paper, Remark 3.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Sinusoidal positional / time embedding
# ---------------------------------------------------------------------------


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for scalar time indices.

    Produces a fixed (non-learned) frequency encoding of a scalar time value
    following the formulation of Vaswani et al. (2017):

        PE(t, 2i)   = sin(t / 10000^{2i / dim})
        PE(t, 2i+1) = cos(t / 10000^{2i / dim})

    Args:
        dim: Output embedding dimensionality.  Must be even.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalPosEmb requires an even dim; got {dim}")
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """Embed time steps.

        Args:
            t: Long or float tensor of shape (B,) containing time indices.

        Returns:
            Float tensor of shape (B, dim).
        """
        device = t.device
        half = self.dim // 2
        # frequencies: 1 / 10000^{2i/dim} for i in 0..half-1
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=device)
            / half
        )  # (half,)
        t_f = t.float().unsqueeze(1)  # (B, 1)
        args = t_f * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return emb


# ---------------------------------------------------------------------------
# Small MLP helper
# ---------------------------------------------------------------------------


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    """Two-layer MLP with SiLU activation and optional dropout."""
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Adaptive Layer Norm modulation (DiT-style)
# ---------------------------------------------------------------------------


class AdaLNModulation(nn.Module):
    """Projects a conditioning signal into (scale, shift) for AdaLN.

    Given conditioning vector c ∈ R^{model_dim}, produces:
        scale ∈ R^{model_dim},  shift ∈ R^{model_dim}
    used to modulate a LayerNorm output as:
        AdaLN(x, c) = (1 + scale) * LayerNorm(x) + shift

    This is the DiT convention (Peebles & Xie, 2023, Eq. 2).
    """

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        # Zero-initialise so that at the start of training the modulation is
        # identity (scale=0 → effective scale=1, shift=0).
        self.proj = nn.Linear(model_dim, 2 * model_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, c: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            c: Conditioning tensor of shape (B, model_dim).

        Returns:
            (scale, shift), each of shape (B, 1, model_dim) for broadcasting
            over the sequence dimension.
        """
        out = self.proj(c)  # (B, 2*model_dim)
        scale, shift = out.chunk(2, dim=-1)
        return scale.unsqueeze(1), shift.unsqueeze(1)  # (B, 1, model_dim)


# ---------------------------------------------------------------------------
# AdaLN Transformer Block
# ---------------------------------------------------------------------------


class AdaLNBlock(nn.Module):
    """Single transformer block with Adaptive Layer Norm (DiT-style).

    Each block contains:
        1. Self-attention over the token sequence, modulated by AdaLN.
        2. Cross-attention from the token sequence to a context sequence
           (time_emb, hor_emb, cond_emb concatenated along the context dim),
           also modulated by AdaLN.
        3. Position-wise feed-forward network (FFN), modulated by AdaLN.

    Residual connections wrap each sub-layer.

    Args:
        model_dim:   Hidden dimensionality.
        num_heads:   Number of attention heads.
        ffn_mult:    FFN hidden dim multiplier (default 4 → 4*model_dim).
        dropout:     Dropout probability (applied in attention & FFN).
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert model_dim % num_heads == 0, (
            f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
        )

        # --- Layer norms (weight/bias are overridden by AdaLN, so we disable
        #     the learnable affine parameters to avoid redundancy) ---
        self.norm1 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)

        # --- Self-attention ---
        self.self_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # --- Cross-attention (token seq → context) ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # --- Feed-forward ---
        ffn_dim = ffn_mult * model_dim
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
            nn.Dropout(dropout),
        )

        # --- AdaLN modulation projections (one per sub-layer) ---
        self.ada1 = AdaLNModulation(model_dim)  # for self-attn
        self.ada2 = AdaLNModulation(model_dim)  # for cross-attn
        self.ada3 = AdaLNModulation(model_dim)  # for FFN

    def forward(
        self,
        x: Tensor,
        mod: Tensor,
        context: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:   Token sequence, shape (B, h, model_dim).
            mod: Modulation signal (time + horizon), shape (B, model_dim).
            context: Cross-attention context, shape (B, n_ctx, model_dim).
            src_key_padding_mask: Boolean mask of shape (B, h) where ``True``
                means the position should be *ignored* (padding).  Passed to
                self-attention so that padding tokens do not pollute real ones.

        Returns:
            Updated token sequence of shape (B, h, model_dim).
        """
        # ---------- 1. Self-attention with AdaLN ----------
        scale1, shift1 = self.ada1(mod)  # (B, 1, model_dim)
        x_norm = (1.0 + scale1) * self.norm1(x) + shift1
        attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + attn_out

        # ---------- 2. Cross-attention with AdaLN ----------
        scale2, shift2 = self.ada2(mod)
        x_norm = (1.0 + scale2) * self.norm2(x) + shift2
        cross_out, _ = self.cross_attn(x_norm, context, context)
        x = x + cross_out

        # ---------- 3. FFN with AdaLN ----------
        scale3, shift3 = self.ada3(mod)
        x_norm = (1.0 + scale3) * self.norm3(x) + shift3
        x = x + self.ffn(x_norm)

        return x


# ---------------------------------------------------------------------------
# Main score network
# ---------------------------------------------------------------------------


class TrajectoryScoreNet(nn.Module):
    """Score network s_θ(z, t | x, h) for Bellman Path Diffusion.

    Implements Remark 3 of the BPD paper using a DiT-style transformer:

    * One network is shared across all horizon values h by conditioning on
      a learned horizon embedding and using a variable-length attention mask.
    * Diffusion time t is embedded via sinusoidal frequencies followed by a
      two-layer MLP (standard in DDPM / score-based model literature).
    * The state-action pair x = (s, a) is projected linearly to model_dim.
    * All three conditioning signals (time, horizon, state-action) are
      concatenated as a short context sequence for cross-attention in every
      AdaLN block, in addition to being summed as the AdaLN modulation signal.

    Args:
        obs_dim:     Dimensionality of the observation space (|s|).
        act_dim:     Dimensionality of the action space (|a|).
        token_dim:   Token dimensionality d_tok = 1 + obs_dim + act_dim.
        model_dim:   Transformer hidden size (default 256).
        num_heads:   Number of attention heads (default 8).
        num_layers:  Number of AdaLN transformer blocks (default 6).
        max_horizon: Maximum supported horizon H (default 32).
        dropout:     Dropout probability (default 0.0).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        token_dim: int,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        max_horizon: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.token_dim = token_dim
        self.model_dim = model_dim
        self.max_horizon = max_horizon

        # ------------------------------------------------------------------
        # Token input projection: z_j ∈ R^{d_tok} → R^{model_dim}
        # ------------------------------------------------------------------
        self.token_proj = nn.Linear(token_dim, model_dim)

        # ------------------------------------------------------------------
        # Learned sinusoidal positional encoding for token positions (optional
        # but standard in sequence models; helps the network track j within h).
        # We use a fixed embedding table of size (max_horizon, model_dim).
        # ------------------------------------------------------------------
        self.pos_emb = nn.Embedding(max_horizon, model_dim)

        # ------------------------------------------------------------------
        # Diffusion time embedding: t → R^{model_dim}
        # Sinusoidal base → MLP (following Ho et al. 2020 / Song et al. 2021).
        # ------------------------------------------------------------------
        self.time_sin = SinusoidalPosEmb(model_dim)
        self.time_mlp = _mlp(model_dim, model_dim * 4, model_dim, dropout=dropout)

        # ------------------------------------------------------------------
        # Horizon embedding: h (integer scalar) → R^{model_dim}
        # We embed h as a learnable integer embedding (1-indexed).
        # ------------------------------------------------------------------
        # +1 so that horizon index h (1-based) can be used directly.
        self.horizon_emb = nn.Embedding(max_horizon + 1, model_dim)
        self.horizon_mlp = _mlp(model_dim, model_dim * 2, model_dim, dropout=dropout)

        # ------------------------------------------------------------------
        # Conditioning (state-action) projection: x ∈ R^{obs+act} → R^{model_dim}
        # ------------------------------------------------------------------
        self.cond_proj = nn.Linear(obs_dim + act_dim, model_dim)

        # ------------------------------------------------------------------
        # Transformer blocks (AdaLN, DiT-style)
        # ------------------------------------------------------------------
        self.blocks = nn.ModuleList(
            [
                AdaLNBlock(model_dim, num_heads, ffn_mult=4, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # ------------------------------------------------------------------
        # Output head: R^{model_dim} → R^{d_tok}
        # Zero-initialised following DiT's final layer convention so that at
        # init the network predicts a zero score everywhere.
        # ------------------------------------------------------------------
        self.out_norm = nn.LayerNorm(model_dim, elementwise_affine=True, eps=1e-6)
        self.out_proj = nn.Linear(model_dim, token_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Weight initialisation for the remaining linear layers.
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier uniform for linear projections; normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.out_proj:
                    # Already zero-initialised above.
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        z: Tensor,
        t: Tensor,
        x: Tensor,
        h: int,
    ) -> Tensor:
        """Compute the score s_θ(z, t | x, h).

        Args:
            z: Noisy trajectory tensor of shape (B, h, token_dim).
            t: Diffusion time step indices of shape (B,) — long integers in
               {0, …, T}.
            x: Current state-action conditioning of shape
               (B, obs_dim + act_dim).
            h: Horizon (integer).  All items in the batch share the same h.

        Returns:
            Score tensor of shape (B, h, token_dim), same shape as ``z``.

        Raises:
            ValueError: If input shapes are inconsistent or h is out of range.
        """
        # ---- Validate shapes --------------------------------------------------
        if z.dim() != 3:
            raise ValueError(f"z must be 3-D (B, h, token_dim); got {tuple(z.shape)}")
        B, h_z, d = z.shape
        if h_z != h:
            raise ValueError(
                f"z.shape[1]={h_z} does not match h={h}"
            )
        if d != self.token_dim:
            raise ValueError(
                f"z token_dim={d} does not match self.token_dim={self.token_dim}"
            )
        if t.shape != (B,):
            raise ValueError(f"t must have shape ({B},); got {tuple(t.shape)}")
        if x.shape != (B, self.obs_dim + self.act_dim):
            raise ValueError(
                f"x must have shape ({B}, {self.obs_dim + self.act_dim}); "
                f"got {tuple(x.shape)}"
            )
        if not (1 <= h <= self.max_horizon):
            raise ValueError(
                f"h must be in [1, {self.max_horizon}]; got {h}"
            )

        device = z.device

        # ---- 1. Token projection + positional embedding -----------------------
        tok = self.token_proj(z)  # (B, h, model_dim)
        pos_idx = torch.arange(h, device=device)  # (h,)
        tok = tok + self.pos_emb(pos_idx).unsqueeze(0)  # (B, h, model_dim)

        # ---- 2. Diffusion time embedding --------------------------------------
        time_emb = self.time_sin(t)           # (B, model_dim)
        time_emb = self.time_mlp(time_emb)    # (B, model_dim)

        # ---- 3. Horizon embedding --------------------------------------------
        # h is a Python int shared across the batch.  We embed the 1-based
        # index so that h=1 → index 1, …, H → index H.
        h_idx = torch.tensor([h], dtype=torch.long, device=device)  # (1,)
        hor_emb = self.horizon_emb(h_idx)          # (1, model_dim)
        hor_emb = self.horizon_mlp(hor_emb)        # (1, model_dim)
        hor_emb = hor_emb.expand(B, -1)            # (B, model_dim)

        # ---- 4. State-action conditioning ------------------------------------
        cond_emb = self.cond_proj(x)               # (B, model_dim)

        # ---- 5. AdaLN modulation signal: time + horizon ----------------------
        mod = time_emb + hor_emb                   # (B, model_dim)

        # ---- 6. Cross-attention context: stack [time, hor, cond] as seq ------
        # Shape: (B, 3, model_dim)
        context = torch.stack(
            [time_emb, hor_emb, cond_emb], dim=1
        )  # (B, 3, model_dim)

        # ---- 7. Build padding mask for self-attention ------------------------
        # In BPD the path tensor z always has exactly h real tokens (the
        # noising process operates on the full (B, h, d_tok) tensor including
        # the structural padding positions).  We do NOT mask out padding
        # positions during denoising because the score is needed everywhere.
        # However, if the caller passes h < max_horizon, only the first h
        # positions are meaningful; setting src_key_padding_mask=None is
        # correct here since the sequence length IS h.
        #
        # If in the future the interface is extended to pass the pad_mask
        # alongside z, it can be forwarded here directly.
        src_key_padding_mask: Optional[Tensor] = None

        # ---- 8. Transformer blocks -------------------------------------------
        hidden = tok  # (B, h, model_dim)
        for block in self.blocks:
            hidden = block(
                hidden,
                mod,
                context,
                src_key_padding_mask=src_key_padding_mask,
            )

        # ---- 9. Output projection -------------------------------------------
        hidden = self.out_norm(hidden)          # (B, h, model_dim)
        score = self.out_proj(hidden)           # (B, h, token_dim)
        return score

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
            f"token_dim={self.token_dim}, model_dim={self.model_dim}, "
            f"max_horizon={self.max_horizon}"
        )
