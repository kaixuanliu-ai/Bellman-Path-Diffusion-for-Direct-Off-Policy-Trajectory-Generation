"""
Trajectory path constructions for Bellman Path Diffusion (BPD).

Implements the path algebra from:
  "Bellman Path Diffusion for Direct Off-Policy Trajectory Generation"

Key equations:
  - Token y = (r, s', a')  ∈  Y = R × S × A
  - Padding token ⊥  ↦  zeros(d_tok)  (tracked via a separate boolean mask)
  - Path w = (w_0, ..., w_{h-1})  ∈  W_h
  - stop_h(y)    = (y, ⊥, ..., ⊥)          [Eq. 4]
  - push_h(y, w) = (y, w_0, ..., w_{h-2})  [Eq. 5]
  - Path length  L_h(w) = number of non-padding tokens
  - Length law:  Pr(L_h = l) = (1-γ)γ^{l-1}  for 1 ≤ l < h;
                               γ^{h-1}         for l = h  [Eq. 14]
  - Value identity: E[Σ_{t<L_h} R_t] = E_π[Σ_{t=0}^{h-1} γ^t R_t | X_0]
                                                              [Eq. 15]

Token embedding layout (each row of the path tensor):
  [r  |  s'_0 ... s'_{obs_dim-1}  |  a'_0 ... a'_{act_dim-1}]
   ^1          ^obs_dim                       ^act_dim

Shapes used throughout:
  d_tok = 1 + obs_dim + act_dim
  path tensor w  : (h, d_tok)   – float32
  pad mask       : (h,)         – bool, True where token is padding (⊥)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Token encoding / decoding
# ---------------------------------------------------------------------------


def token_dim(obs_dim: int, act_dim: int) -> int:
    """Return d_tok = 1 + obs_dim + act_dim."""
    return 1 + obs_dim + act_dim


def encode_token(
    reward: float | Tensor,
    next_obs: Tensor,
    next_action: Tensor,
) -> Tensor:
    """Encode a single transition into a token vector y ∈ R^{d_tok}.

    Args:
        reward:      Scalar reward r (Python float or 0-d / 1-d Tensor).
        next_obs:    Next observation s' of shape (obs_dim,).
        next_action: Next action a' of shape (act_dim,).

    Returns:
        Token tensor of shape (d_tok,) = (1 + obs_dim + act_dim,).
    """
    if not isinstance(reward, Tensor):
        reward = torch.tensor(reward, dtype=torch.float32)
    reward = reward.reshape(1).to(dtype=torch.float32)
    next_obs = next_obs.reshape(-1).to(dtype=torch.float32)
    next_action = next_action.reshape(-1).to(dtype=torch.float32)
    return torch.cat([reward, next_obs, next_action], dim=0)


def encode_transition(
    transition: Dict[str, object],
    obs_dim: int,
    act_dim: int,
) -> Tuple[Tensor, int, int]:
    """Encode a raw transition dictionary into a token tensor.

    Expected keys in *transition*:
        ``"reward"``      – float or Tensor
        ``"next_obs"``    – array-like of shape (obs_dim,)
        ``"next_action"`` – array-like of shape (act_dim,)

    Args:
        transition: Dictionary with keys ``"reward"``, ``"next_obs"``,
                    ``"next_action"``.
        obs_dim:    Dimensionality of the observation space.
        act_dim:    Dimensionality of the action space.

    Returns:
        Tuple ``(token, obs_dim, act_dim)`` where ``token`` has shape
        ``(d_tok,)``.
    """
    reward = transition["reward"]
    next_obs = torch.as_tensor(transition["next_obs"], dtype=torch.float32)
    next_action = torch.as_tensor(transition["next_action"], dtype=torch.float32)

    if next_obs.shape != (obs_dim,):
        raise ValueError(
            f"next_obs must have shape ({obs_dim},); got {tuple(next_obs.shape)}"
        )
    if next_action.shape != (act_dim,):
        raise ValueError(
            f"next_action must have shape ({act_dim},); got {tuple(next_action.shape)}"
        )

    token = encode_token(reward, next_obs, next_action)
    return token, obs_dim, act_dim


def decode_token(
    token: Tensor,
    obs_dim: int,
    act_dim: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Split a token vector back into (reward, next_obs, next_action).

    Args:
        token:   Shape (d_tok,).
        obs_dim: Dimensionality of the observation space.
        act_dim: Dimensionality of the action space.

    Returns:
        Tuple ``(reward, next_obs, next_action)`` with shapes
        ``(1,)``, ``(obs_dim,)``, ``(act_dim,)``.
    """
    d_tok = 1 + obs_dim + act_dim
    if token.shape != (d_tok,):
        raise ValueError(
            f"token must have shape ({d_tok},); got {tuple(token.shape)}"
        )
    reward = token[0:1]
    next_obs = token[1 : 1 + obs_dim]
    next_action = token[1 + obs_dim :]
    return reward, next_obs, next_action


def padding_token(d_tok: int, device: torch.device | None = None) -> Tensor:
    """Return the padding token φ(⊥) = zeros(d_tok).

    Args:
        d_tok:  Token dimensionality.
        device: Target device.

    Returns:
        Zero tensor of shape (d_tok,).
    """
    return torch.zeros(d_tok, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Path constructions  (Equations 4 and 5)
# ---------------------------------------------------------------------------


def stop_map(
    y: Tensor,
    h: int,
) -> Tuple[Tensor, Tensor]:
    """stop_h(y) = (y, ⊥, ..., ⊥)  [Eq. 4].

    Places token *y* in position 0 and fills the remaining h-1 positions with
    the padding token.

    Args:
        y: Token tensor of shape (d_tok,).
        h: Path horizon (number of steps).

    Returns:
        Tuple ``(w, pad_mask)`` where

        * ``w``        has shape ``(h, d_tok)`` – the path tensor.
        * ``pad_mask`` has shape ``(h,)``        – ``True`` at padding positions.
    """
    if y.dim() != 1:
        raise ValueError(f"y must be 1-D; got shape {tuple(y.shape)}")
    d_tok = y.shape[0]
    if h < 1:
        raise ValueError(f"h must be >= 1; got {h}")

    w = torch.zeros(h, d_tok, dtype=torch.float32, device=y.device)
    w[0] = y.to(dtype=torch.float32)

    pad_mask = torch.ones(h, dtype=torch.bool, device=y.device)
    pad_mask[0] = False  # position 0 is a real token

    return w, pad_mask


def push_map(
    y: Tensor,
    w_prev: Tensor,
    pad_mask_prev: Tensor,
) -> Tuple[Tensor, Tensor]:
    """push_h(y, w) = (y, w_0, ..., w_{h-2})  [Eq. 5].

    Prepends token *y* to the path *w_prev* of length h-1, dropping the last
    token of *w_prev* (which must be a padding token by construction of the
    path distribution).

    Args:
        y:            Token tensor of shape (d_tok,).
        w_prev:       Path tensor of shape (h-1, d_tok).
        pad_mask_prev: Boolean mask of shape (h-1,); ``True`` where padding.

    Returns:
        Tuple ``(w, pad_mask)`` where

        * ``w``        has shape ``(h, d_tok)``.
        * ``pad_mask`` has shape ``(h,)``.
    """
    if y.dim() != 1:
        raise ValueError(f"y must be 1-D; got shape {tuple(y.shape)}")
    if w_prev.dim() != 2:
        raise ValueError(
            f"w_prev must be 2-D (h-1, d_tok); got shape {tuple(w_prev.shape)}"
        )
    h_minus_1, d_tok = w_prev.shape
    if y.shape[0] != d_tok:
        raise ValueError(
            f"y dim ({y.shape[0]}) must match w_prev d_tok ({d_tok})"
        )
    if pad_mask_prev.shape != (h_minus_1,):
        raise ValueError(
            f"pad_mask_prev must have shape ({h_minus_1},); "
            f"got {tuple(pad_mask_prev.shape)}"
        )

    h = h_minus_1 + 1

    # New path: [y, w_0, ..., w_{h-2}]  (drop the last row of w_prev)
    w = torch.zeros(h, d_tok, dtype=torch.float32, device=y.device)
    w[0] = y.to(dtype=torch.float32)
    w[1:] = w_prev[:h_minus_1].to(dtype=torch.float32, device=y.device)

    # New mask: [False, mask_0, ..., mask_{h-3}]  (drop last mask entry)
    pad_mask = torch.zeros(h, dtype=torch.bool, device=y.device)
    pad_mask[0] = False  # y is a real token
    pad_mask[1:] = pad_mask_prev[:h_minus_1]

    return w, pad_mask


# ---------------------------------------------------------------------------
# Path statistics
# ---------------------------------------------------------------------------


def path_length(pad_mask: Tensor) -> int:
    """L_h(w) = number of non-padding tokens in the path.

    Args:
        pad_mask: Boolean tensor of shape (h,); ``True`` at padding positions.

    Returns:
        Integer count of real (non-padding) tokens.
    """
    if pad_mask.dim() != 1:
        raise ValueError(
            f"pad_mask must be 1-D; got shape {tuple(pad_mask.shape)}"
        )
    return int((~pad_mask).sum().item())


def compute_return(
    w: Tensor,
    pad_mask: Tensor,
) -> Tensor:
    """Σ_{t < L_h(w)} R_t  – undiscounted sum of rewards over real tokens.

    Rewards are stored in the first element (index 0) of each token vector.
    Padding tokens are excluded from the sum.

    Args:
        w:        Path tensor of shape (h, d_tok).
        pad_mask: Boolean mask of shape (h,); ``True`` at padding positions.

    Returns:
        Scalar tensor (shape ``()``) with the cumulative reward.
    """
    if w.dim() != 2:
        raise ValueError(f"w must be 2-D; got shape {tuple(w.shape)}")
    if pad_mask.shape != (w.shape[0],):
        raise ValueError(
            f"pad_mask must have shape ({w.shape[0]},); "
            f"got {tuple(pad_mask.shape)}"
        )
    # Mask out padding rows before extracting rewards.
    real_mask = ~pad_mask  # (h,)
    rewards = w[:, 0]  # (h,)
    return (rewards * real_mask.to(dtype=rewards.dtype)).sum()


def compute_discounted_return(
    w: Tensor,
    pad_mask: Tensor,
    gamma: float,
) -> Tensor:
    """Σ_{t < L_h(w)} γ^t R_t  – discounted return over real tokens.

    Provides the left-hand side of the value identity [Eq. 15]:

        E[Σ_{t < L_h} γ^t R_t] = E_π[Σ_{t=0}^{h-1} γ^t R_t | X_0]

    Args:
        w:        Path tensor of shape (h, d_tok).
        pad_mask: Boolean mask of shape (h,); ``True`` at padding positions.
        gamma:    Discount factor γ ∈ (0, 1].

    Returns:
        Scalar tensor with the discounted cumulative reward.
    """
    if w.dim() != 2:
        raise ValueError(f"w must be 2-D; got shape {tuple(w.shape)}")
    if pad_mask.shape != (w.shape[0],):
        raise ValueError(
            f"pad_mask must have shape ({w.shape[0]},); "
            f"got {tuple(pad_mask.shape)}"
        )
    h = w.shape[0]
    # γ^0, γ^1, ..., γ^{h-1}
    discounts = torch.tensor(
        [gamma ** t for t in range(h)],
        dtype=w.dtype,
        device=w.device,
    )
    real_mask = (~pad_mask).to(dtype=w.dtype)
    rewards = w[:, 0]
    return (rewards * discounts * real_mask).sum()


# ---------------------------------------------------------------------------
# Length distribution  (Eq. 14)
# ---------------------------------------------------------------------------


def length_log_prob(l: int, h: int, gamma: float) -> Tensor:
    """Log-probability of path length L_h = l under the geometric law [Eq. 14].

        Pr(L_h = l) = (1 - γ) γ^{l-1}   for 1 ≤ l < h
        Pr(L_h = h) = γ^{h-1}

    Args:
        l:     Observed path length (1 ≤ l ≤ h).
        h:     Path horizon.
        gamma: Discount factor γ ∈ (0, 1).

    Returns:
        Scalar log-probability tensor.
    """
    if not (1 <= l <= h):
        raise ValueError(f"l must satisfy 1 ≤ l ≤ h={h}; got l={l}")
    if not (0.0 < gamma < 1.0):
        raise ValueError(f"gamma must be in (0, 1); got {gamma}")

    if l < h:
        log_prob = torch.log(torch.tensor((1.0 - gamma) * gamma ** (l - 1)))
    else:
        # l == h
        log_prob = torch.tensor((h - 1) * float(torch.log(torch.tensor(gamma)).item()))

    return log_prob


def sample_path_length(h: int, gamma: float, device: torch.device | None = None) -> int:
    """Draw a random path length l ~ Pr(L_h = ·)  [Eq. 14].

    The geometric distribution is truncated at h:

        Pr(L_h = l) = (1 - γ) γ^{l-1}   for 1 ≤ l < h
        Pr(L_h = h) = γ^{h-1}

    Args:
        h:      Path horizon (maximum length).
        gamma:  Discount factor γ ∈ (0, 1).
        device: Torch device (unused; returned value is a Python int).

    Returns:
        Sampled length l as a Python int in {1, ..., h}.
    """
    if not (0.0 < gamma < 1.0):
        raise ValueError(f"gamma must be in (0, 1); got {gamma}")
    if h < 1:
        raise ValueError(f"h must be >= 1; got {h}")

    probs = torch.zeros(h, dtype=torch.float64)
    for l in range(1, h):
        probs[l - 1] = (1.0 - gamma) * gamma ** (l - 1)
    probs[h - 1] = gamma ** (h - 1)

    idx = torch.multinomial(probs.float(), num_samples=1).item()
    return int(idx) + 1  # convert 0-based index to 1-based length


# ---------------------------------------------------------------------------
# Path construction from a trajectory rollout
# ---------------------------------------------------------------------------


def build_path_from_rollout(
    transitions: list[Dict[str, object]],
    obs_dim: int,
    act_dim: int,
    h: int,
) -> Tuple[Tensor, Tensor]:
    """Build a path tensor from a list of transition dictionaries.

    Each dictionary must contain keys ``"reward"``, ``"next_obs"``,
    ``"next_action"``.  The list is treated as an ordered sequence
    (w_0, w_1, ...).  If ``len(transitions) < h``, the remaining slots are
    filled with padding tokens.  If ``len(transitions) > h``, the list is
    silently truncated to the first *h* entries.

    Args:
        transitions: Sequence of transition dicts (each with ``"reward"``,
                     ``"next_obs"``, ``"next_action"``).
        obs_dim:     Observation dimensionality.
        act_dim:     Action dimensionality.
        h:           Path horizon.

    Returns:
        Tuple ``(w, pad_mask)`` where

        * ``w``        has shape ``(h, d_tok)``.
        * ``pad_mask`` has shape ``(h,)`` – ``True`` at padding positions.
    """
    d_tok = token_dim(obs_dim, act_dim)
    w = torch.zeros(h, d_tok, dtype=torch.float32)
    pad_mask = torch.ones(h, dtype=torch.bool)

    for t, trans in enumerate(transitions[:h]):
        token, _, _ = encode_transition(trans, obs_dim, act_dim)
        w[t] = token
        pad_mask[t] = False

    return w, pad_mask
