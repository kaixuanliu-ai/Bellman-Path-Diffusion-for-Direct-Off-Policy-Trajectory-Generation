"""
Suffix Replay Buffer for Bellman Path Diffusion.

Implements the Replay-Buffered Noisy Suffix Sampling scheme from Section 6 of
"Bellman Path Diffusion for Direct Off-Policy Trajectory Generation".

Key concepts:
  - A replay entry stores (x', W_+) where x' = (s', a') is the successor pair
    and W_+ is a CLEAN teacher-generated suffix of length (h-1) [Eq. 49].
  - Proposition 1 (Replay Equivalence): corrupting a cached clean suffix with
    the blockwise kernel at time t gives exactly the same population marginal
    as partial reverse integration of the teacher SDE to time t [Eq. 50].
  - Remark 8: the cached suffix MUST remain attached to its sampled x'. For
    continuous conditioning, nearest-neighbour reuse is an approximation and
    is NOT used here; this implementation enforces exact match per the paper.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional, Tuple

from torch import Tensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_key(x_prime: Tensor, decimals: Optional[int] = None) -> Tuple:
    """Convert a successor-pair tensor to a hashable cache key.

    By default no rounding is performed: replay is keyed by the exact float32
    successor pair required by Section 6.  Passing ``decimals`` explicitly
    enables an approximate discretized replay variant, which is outside the
    paper's exact replay-equivalence statement.

    Args:
        x_prime: 1-D tensor of shape ``(state_dim + act_dim,)``.
        decimals: Optional number of decimal places used for approximate keys.

    Returns:
        A plain Python tuple that can be used as a dict key.
    """
    if x_prime.dim() != 1:
        raise ValueError(f"x_prime must be 1-D, got shape {tuple(x_prime.shape)}")
    values = x_prime.detach().cpu().float().numpy()
    if decimals is not None:
        values = values.round(decimals)
    return tuple(values.tolist())


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------


class SuffixReplayBuffer:
    """Thread-safe fixed-capacity buffer that maps each x' to its clean suffix.

    Each entry is the pair (x', W_+) from Eq. 49 of the paper:
      - ``x'`` — the successor observation-action pair ``(s', a')``.
      - ``W_+`` — a clean (noise-free) suffix trajectory of length
        ``suffix_horizon``, with token dimension ``token_dim``.

    When the buffer is full the oldest entry is evicted (LRU-style FIFO via
    ``collections.OrderedDict``).

    Thread safety is provided by a single ``threading.Lock``; all public
    methods acquire the lock before touching shared state.

    Args:
        max_size: Maximum number of (x', W_+) pairs to retain.
        suffix_horizon: Number of suffix steps h-1 stored per entry.
        token_dim: Dimensionality of each step in the suffix (e.g. state+act).
        key_decimals: ``None`` (default) preserves exact float32 keys.  An
            integer opts into approximate rounded keys and therefore gives up
            the paper's exact conditional replay guarantee.
    """

    def __init__(
        self,
        max_size: int,
        suffix_horizon: int,
        token_dim: int,
        key_decimals: Optional[int] = None,
    ) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if suffix_horizon <= 0:
            raise ValueError(f"suffix_horizon must be positive, got {suffix_horizon}")
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}")

        self._max_size: int = max_size
        self._suffix_horizon: int = suffix_horizon
        self._token_dim: int = token_dim
        self._key_decimals: Optional[int] = key_decimals

        # OrderedDict preserves insertion order so oldest entries are evicted
        # first when the buffer overflows.
        self._store: OrderedDict[Tuple, Tensor] = OrderedDict()
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(self, x_prime: Tensor, clean_suffix: Tensor) -> None:
        """Store a (x', W_+) entry in the buffer.

        If an entry for this x' already exists it is updated in-place and its
        recency is refreshed (moved to end of the eviction queue).  If the
        buffer is at capacity the oldest entry is evicted before insertion.

        Args:
            x_prime: Successor pair tensor of shape ``(state_dim + act_dim,)``.
            clean_suffix: Clean teacher-generated suffix of shape
                ``(suffix_horizon, token_dim)``.

        Raises:
            ValueError: If ``clean_suffix`` has an unexpected shape.
        """
        if clean_suffix.shape != (self._suffix_horizon, self._token_dim):
            raise ValueError(
                f"clean_suffix must have shape "
                f"({self._suffix_horizon}, {self._token_dim}), "
                f"got {tuple(clean_suffix.shape)}"
            )

        key = _to_key(x_prime, self._key_decimals)
        # Store a detached CPU copy so the buffer does not hold gradient graphs
        # or references to GPU memory.
        suffix_stored = clean_suffix.detach().cpu().clone()

        with self._lock:
            if key in self._store:
                # Refresh recency: remove and re-insert at end.
                del self._store[key]
            elif len(self._store) >= self._max_size:
                # Evict oldest (first) entry.
                self._store.popitem(last=False)
            self._store[key] = suffix_stored

    def sample(self, x_prime: Tensor) -> Tensor:
        """Retrieve the stored suffix for an EXACT x' match.

        Per Remark 8 of the paper, the cached suffix must remain attached to
        its own sampled x'.  With the default configuration only an exact
        float32 key match is accepted; nearest-neighbour reuse is intentionally
        not implemented.

        Args:
            x_prime: Successor pair tensor of shape ``(state_dim + act_dim,)``.

        Returns:
            The stored clean suffix tensor of shape
            ``(suffix_horizon, token_dim)``, on CPU.

        Raises:
            KeyError: If no entry exists for this x'.
        """
        key = _to_key(x_prime, self._key_decimals)
        with self._lock:
            if key not in self._store:
                raise KeyError(
                    f"No suffix found for x' with key {key}. "
                    "Per Remark 8, only exact matches are accepted."
                )
            return self._store[key].clone()

    def sample_or_none(self, x_prime: Tensor) -> Optional[Tensor]:
        """Retrieve the stored suffix, or ``None`` if not found.

        Convenience wrapper around :meth:`sample` that swallows the
        ``KeyError`` and returns ``None`` instead.  Useful when callers prefer
        a conditional branch over exception handling.

        Args:
            x_prime: Successor pair tensor of shape ``(state_dim + act_dim,)``.

        Returns:
            The stored clean suffix tensor, or ``None`` if absent.
        """
        try:
            return self.sample(x_prime)
        except KeyError:
            return None

    def clear(self) -> None:
        """Remove all entries from the buffer."""
        with self._lock:
            self._store.clear()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the current number of stored entries."""
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"size={len(self)}/{self._max_size}, "
            f"suffix_horizon={self._suffix_horizon}, "
            f"token_dim={self._token_dim})"
        )
