"""Shared-horizon trajectory diffusion transformer.

The network follows the adaLN-Zero organization of DiT (Peebles & Xie,
ICCV 2023), adapted from image patches to trajectory-token blocks.  In the
paper's discrete DDPM parameterization it predicts epsilon; the corresponding
score is ``-epsilon / sigma_t``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal diffusion-index embedding followed by a two-layer MLP."""

    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half, 1)
        )
        arguments = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((arguments.cos(), arguments.sin()), dim=-1)
        if dim % 2:
            embedding = torch.cat(
                (embedding, embedding.new_zeros(t.shape[0], 1)), dim=-1
            )
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.sinusoidal(t, self.frequency_size))


class AdaLNZeroBlock(nn.Module):
    """Pre-norm transformer block with gated adaLN-Zero conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(condition).chunk(6, dim=-1)
        )
        normalized = modulate(self.norm1(tokens), shift_attn, scale_attn)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + gate_attn.unsqueeze(1) * attended
        tokens = tokens + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(tokens), shift_mlp, scale_mlp)
        )
        return tokens


class FinalLayer(nn.Module):
    """Conditioned final normalization and token-wise epsilon projection."""

    def __init__(self, hidden_size: int, token_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.projection = nn.Linear(hidden_size, token_dim)

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.projection(modulate(self.norm(tokens), shift, scale))


class TrajectoryScoreNet(nn.Module):
    """Predict DDPM noise for a noisy reward-bearing path.

    A single model handles ``h=1,...,H``.  Calls are homogeneous in horizon,
    so the sequence is physically sliced to ``h`` rather than padded; this is
    equivalent to the variable-length attention mask described in Remark 3.

    Signature: ``network(z_t, t, x, h) -> epsilon``.
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
        if token_dim != 2 + obs_dim + act_dim:
            raise ValueError(
                "token_dim must equal 2 + obs_dim + act_dim "
                "(reward + obs + act + injective-phi flag)"
            )
        if max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.token_dim = int(token_dim)
        self.model_dim = int(model_dim)
        self.max_horizon = int(max_horizon)

        self.token_projection = nn.Linear(token_dim, model_dim)
        self.position_embedding = nn.Embedding(max_horizon, model_dim)
        self.time_embedding = TimestepEmbedder(model_dim)
        self.horizon_embedding = nn.Embedding(max_horizon + 1, model_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(obs_dim + act_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.blocks = nn.ModuleList(
            AdaLNZeroBlock(model_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        )
        self.final_layer = FinalLayer(model_dim, token_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Use DiT's stable adaLN-Zero initialization pattern."""

        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.horizon_embedding.weight, std=0.02)
        for linear in (self.time_embedding.mlp[0], self.time_embedding.mlp[2]):
            nn.init.normal_(linear.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.modulation[-1].weight)
        nn.init.zeros_(self.final_layer.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.projection.weight)
        nn.init.zeros_(self.final_layer.projection.bias)

    def forward(self, z_t: Tensor, t: Tensor, x: Tensor, h: int) -> Tensor:
        if z_t.ndim != 3:
            raise ValueError(f"z_t must have shape (B, h, d), got {tuple(z_t.shape)}")
        batch_size, sequence_length, token_dim = z_t.shape
        if sequence_length != h:
            raise ValueError(f"sequence length {sequence_length} does not equal h={h}")
        if token_dim != self.token_dim:
            raise ValueError(f"token dimension {token_dim} != {self.token_dim}")
        if not 1 <= h <= self.max_horizon:
            raise ValueError(f"h must lie in [1, {self.max_horizon}], got {h}")
        if t.shape != (batch_size,):
            raise ValueError(f"t must have shape ({batch_size},)")
        expected_condition = (batch_size, self.obs_dim + self.act_dim)
        if x.shape != expected_condition:
            raise ValueError(
                f"x must have shape {expected_condition}, got {tuple(x.shape)}"
            )

        positions = torch.arange(h, device=z_t.device)
        hidden = self.token_projection(z_t) + self.position_embedding(
            positions
        ).unsqueeze(0)
        horizon_index = torch.full(
            (batch_size,), h, dtype=torch.long, device=z_t.device
        )
        condition = (
            self.time_embedding(t)
            + self.horizon_embedding(horizon_index)
            + self.condition_projection(x)
        )
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.final_layer(hidden, condition)

    def num_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def extra_repr(self) -> str:
        return (
            f"obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
            f"token_dim={self.token_dim}, max_horizon={self.max_horizon}"
        )
