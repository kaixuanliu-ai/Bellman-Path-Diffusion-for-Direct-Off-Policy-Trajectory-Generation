"""
Per-feature normalization for offline RL datasets.

Inspired by diffuser/datasets/normalization.py (Janner et al., 2022, NeurIPS).

This module is intentionally free of D4RL or any dataset-loader dependency:
all normalizers are constructed from raw NumPy arrays so they can be used
with any offline dataset pipeline.

Supported normalizers
---------------------
* :class:`Normalizer`         – abstract base class
* :class:`LimitsNormalizer`   – scales each feature to ``[-1, 1]`` using
                                 per-dimension dataset min/max
* :class:`GaussianNormalizer` – zero-mean unit-variance per feature
* :class:`RewardNormalizer`   – thin wrapper around a scalar reward signal
                                 with optional hard clipping

Typical usage::

    data = {
        "observations": obs_array,   # (N, obs_dim)
        "actions":      act_array,   # (N, act_dim)
        "rewards":      rew_array,   # (N,) or (N, 1)
    }

    obs_norm = LimitsNormalizer.fit(data, key="observations")
    act_norm = LimitsNormalizer.fit(data, key="actions")
    rew_norm = RewardNormalizer.fit(data, clip=(-10.0, 10.0))

    obs_n = obs_norm.normalize(obs_array, key="observations")
    obs_r = obs_norm.unnormalize(obs_n,   key="observations")
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Array = np.ndarray
DataDict = Dict[str, Array]

# Keys recognised by the fit() class methods when given a full dataset dict.
SUPPORTED_KEYS: Tuple[str, ...] = ("observations", "actions", "rewards")


# ---------------------------------------------------------------------------
# Helper: safe statistics
# ---------------------------------------------------------------------------


def _safe_std(x: Array, eps: float = 1e-8) -> Array:
    """Per-column standard deviation, clamped away from zero."""
    std = x.std(axis=0)
    return np.where(std < eps, np.ones_like(std), std)


def _safe_range(x: Array, eps: float = 1e-8) -> Tuple[Array, Array]:
    """Per-column (min, max) with a minimum range of *eps*."""
    lo = x.min(axis=0)
    hi = x.max(axis=0)
    # Where a feature is constant, expand range symmetrically around the mean.
    degenerate = (hi - lo) < eps
    midpoint = (hi + lo) / 2.0
    lo = np.where(degenerate, midpoint - 0.5, lo)
    hi = np.where(degenerate, midpoint + 0.5, hi)
    return lo, hi


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Normalizer:
    """Abstract base class for per-feature normalizers.

    Subclasses must implement :meth:`normalize` and :meth:`unnormalize`.
    The *key* argument is accepted by both methods so that a single
    normalizer instance can potentially handle multiple named fields, but
    concrete subclasses are free to ignore it.

    Attributes:
        stats: Dictionary of fitted statistics.  Populated by
               :meth:`fit` in each concrete subclass.
    """

    stats: Dict[str, Array]

    def __init__(self) -> None:
        self.stats = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def normalize(self, x: Array, key: str = "observations") -> Array:
        """Map raw feature array *x* to a normalised representation.

        Args:
            x:   Input array of shape ``(N, d)`` or ``(d,)``.
            key: Dataset field name (e.g. ``"observations"``).  Concrete
                 subclasses may use this to select the correct statistics.

        Returns:
            Normalised array with the same shape as *x*.
        """
        raise NotImplementedError

    def unnormalize(self, x: Array, key: str = "observations") -> Array:
        """Invert a previous :meth:`normalize` call.

        Args:
            x:   Normalised array of shape ``(N, d)`` or ``(d,)``.
            key: Dataset field name; must match the key used during
                 :meth:`normalize`.

        Returns:
            Reconstructed array in the original feature space, same shape as
            *x*.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def fit(cls, data: DataDict, key: str = "observations") -> "Normalizer":
        """Compute statistics from *data[key]* and return a fitted normalizer.

        Args:
            data: Dictionary with at least the entry identified by *key*.
            key:  Which field to use (``"observations"``, ``"actions"``, or
                  ``"rewards"``).

        Returns:
            A fitted :class:`Normalizer` instance.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Array]:
        """Return a copy of the fitted statistics for checkpointing."""
        return {k: v.copy() for k, v in self.stats.items()}

    def load_state_dict(self, state: Dict[str, Array]) -> None:
        """Restore statistics from a previously saved state dict."""
        self.stats = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}

    def __repr__(self) -> str:
        keys = list(self.stats.keys())
        return f"{self.__class__.__name__}(stats_keys={keys})"


# ---------------------------------------------------------------------------
# LimitsNormalizer
# ---------------------------------------------------------------------------


class LimitsNormalizer(Normalizer):
    """Linearly scales each feature dimension to the interval ``[-1, 1]``.

    Given per-dimension minimum ``lo`` and maximum ``hi`` computed from the
    dataset, the forward map is:

    .. math::
        x_{\\text{norm}} = 2 \\cdot \\frac{x - lo}{hi - lo} - 1

    and the inverse is:

    .. math::
        x = \\frac{(x_{\\text{norm}} + 1)}{2} \\cdot (hi - lo) + lo

    If a feature is constant across the dataset its range is widened
    artificially (see :func:`_safe_range`) so that the mapping remains
    well-defined.

    Attributes:
        stats: Contains ``"lo"`` and ``"hi"`` as 1-D arrays of shape
               ``(d,)``.
    """

    def __init__(self, lo: Array, hi: Array) -> None:
        super().__init__()
        lo = np.asarray(lo, dtype=np.float64).reshape(-1)
        hi = np.asarray(hi, dtype=np.float64).reshape(-1)
        self.stats = {"lo": lo, "hi": hi}

    # ------------------------------------------------------------------
    # Core transforms
    # ------------------------------------------------------------------

    def normalize(self, x: Array, key: str = "observations") -> Array:  # noqa: ARG002
        """Scale *x* so every dimension lies in ``[-1, 1]``.

        Args:
            x:   Array of shape ``(N, d)`` or ``(d,)``.
            key: Ignored; included for API consistency.

        Returns:
            Normalised array in ``[-1, 1]``, same shape as *x*.
        """
        x = np.asarray(x, dtype=np.float64)
        lo, hi = self.stats["lo"], self.stats["hi"]
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    def unnormalize(self, x: Array, key: str = "observations") -> Array:  # noqa: ARG002
        """Invert :meth:`normalize`, mapping ``[-1, 1]`` back to the original range.

        Args:
            x:   Normalised array of shape ``(N, d)`` or ``(d,)``.
            key: Ignored; included for API consistency.

        Returns:
            Reconstructed array in the original feature space.
        """
        x = np.asarray(x, dtype=np.float64)
        lo, hi = self.stats["lo"], self.stats["hi"]
        return (x + 1.0) / 2.0 * (hi - lo) + lo

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def fit(  # type: ignore[override]
        cls,
        data: DataDict,
        key: str = "observations",
    ) -> "LimitsNormalizer":
        """Fit min/max statistics from ``data[key]``.

        Args:
            data: Dataset dictionary.  The value ``data[key]`` must be a
                  NumPy array of shape ``(N, d)`` (or ``(N,)`` for scalars).
            key:  Field to fit.  Must be present in *data*.

        Returns:
            Fitted :class:`LimitsNormalizer`.

        Raises:
            KeyError: If *key* is not found in *data*.
            ValueError: If the array has no samples (``N == 0``).
        """
        arr = _extract_array(data, key)
        lo, hi = _safe_range(arr)
        return cls(lo=lo, hi=hi)


# ---------------------------------------------------------------------------
# GaussianNormalizer
# ---------------------------------------------------------------------------


class GaussianNormalizer(Normalizer):
    """Standardises each feature to zero mean and unit variance.

    .. math::
        x_{\\text{norm}} = \\frac{x - \\mu}{\\sigma + \\epsilon}

    where :math:`\\mu` and :math:`\\sigma` are the per-dimension mean and
    standard deviation computed from the dataset.  Features with near-zero
    variance are assigned :math:`\\sigma = 1` to avoid division by zero (a
    warning is issued when this happens).

    Attributes:
        stats: Contains ``"mean"`` and ``"std"`` as 1-D arrays of shape
               ``(d,)``.
    """

    #: Small constant added to ``std`` during normalization.
    EPS: float = 1e-8

    def __init__(self, mean: Array, std: Array) -> None:
        super().__init__()
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        std = np.asarray(std, dtype=np.float64).reshape(-1)
        self.stats = {"mean": mean, "std": std}

    # ------------------------------------------------------------------
    # Core transforms
    # ------------------------------------------------------------------

    def normalize(self, x: Array, key: str = "observations") -> Array:  # noqa: ARG002
        """Standardise *x* to zero mean and unit variance.

        Args:
            x:   Array of shape ``(N, d)`` or ``(d,)``.
            key: Ignored; included for API consistency.

        Returns:
            Standardised array, same shape as *x*.
        """
        x = np.asarray(x, dtype=np.float64)
        mean, std = self.stats["mean"], self.stats["std"]
        return (x - mean) / (std + self.EPS)

    def unnormalize(self, x: Array, key: str = "observations") -> Array:  # noqa: ARG002
        """Invert :meth:`normalize`.

        Args:
            x:   Standardised array of shape ``(N, d)`` or ``(d,)``.
            key: Ignored; included for API consistency.

        Returns:
            Reconstructed array in the original feature space.
        """
        x = np.asarray(x, dtype=np.float64)
        mean, std = self.stats["mean"], self.stats["std"]
        return x * (std + self.EPS) + mean

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def fit(  # type: ignore[override]
        cls,
        data: DataDict,
        key: str = "observations",
    ) -> "GaussianNormalizer":
        """Compute mean and std from ``data[key]``.

        Args:
            data: Dataset dictionary.  ``data[key]`` must be a NumPy array of
                  shape ``(N, d)`` or ``(N,)``.
            key:  Field to fit.

        Returns:
            Fitted :class:`GaussianNormalizer`.

        Raises:
            KeyError: If *key* is not found in *data*.
            ValueError: If the array has no samples.
        """
        arr = _extract_array(data, key)
        mean = arr.mean(axis=0)
        std = _safe_std(arr)

        # Warn when any feature is effectively constant.
        constant_dims = np.where(arr.std(axis=0) < cls.EPS)[0]
        if constant_dims.size > 0:
            warnings.warn(
                f"GaussianNormalizer.fit: {len(constant_dims)} feature dimension(s) "
                f"have near-zero variance for key='{key}' "
                f"(dims {constant_dims.tolist()[:10]}{'...' if len(constant_dims) > 10 else ''}). "
                "Their std has been set to 1 to avoid division by zero.",
                UserWarning,
                stacklevel=2,
            )

        return cls(mean=mean, std=std)


# ---------------------------------------------------------------------------
# RewardNormalizer
# ---------------------------------------------------------------------------


class RewardNormalizer:
    """Normalizer for a scalar reward signal.

    Wraps a :class:`GaussianNormalizer` or :class:`LimitsNormalizer` for the
    ``"rewards"`` key of a dataset and adds optional hard clipping *after*
    normalisation.  The clip bounds are expressed in the *normalised* space so
    that they are interpretable regardless of the dataset's reward scale.

    Args:
        base_normalizer: A fitted :class:`Normalizer` instance (typically
                         :class:`GaussianNormalizer` or
                         :class:`LimitsNormalizer`) whose statistics cover the
                         scalar reward dimension.
        clip:            If provided, normalised rewards are clipped to this
                         ``(lo, hi)`` interval.  ``None`` disables clipping.
    """

    def __init__(
        self,
        base_normalizer: Normalizer,
        clip: Optional[Tuple[float, float]] = None,
    ) -> None:
        self._norm = base_normalizer
        self._clip = clip

    # ------------------------------------------------------------------
    # Core transforms
    # ------------------------------------------------------------------

    def normalize(self, rewards: Array) -> Array:
        """Normalise raw reward array and apply optional clipping.

        Args:
            rewards: Array of shape ``(N,)`` or ``(N, 1)``.

        Returns:
            Normalised (and optionally clipped) rewards, same shape as
            *rewards*.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        scalar_shape = rewards.shape
        flat = rewards.reshape(-1, 1)
        normed = self._norm.normalize(flat, key="rewards").reshape(scalar_shape)
        if self._clip is not None:
            normed = np.clip(normed, self._clip[0], self._clip[1])
        return normed

    def unnormalize(self, rewards: Array) -> Array:
        """Invert :meth:`normalize` (clipping is *not* reversed).

        Args:
            rewards: Normalised reward array of shape ``(N,)`` or ``(N, 1)``.

        Returns:
            Rewards in the original scale.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        scalar_shape = rewards.shape
        flat = rewards.reshape(-1, 1)
        return self._norm.unnormalize(flat, key="rewards").reshape(scalar_shape)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def clip(self) -> Optional[Tuple[float, float]]:
        """The clipping interval used during :meth:`normalize`, or ``None``."""
        return self._clip

    @property
    def base_normalizer(self) -> Normalizer:
        """The underlying :class:`Normalizer` instance."""
        return self._norm

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        data: DataDict,
        normalizer_cls: type = GaussianNormalizer,
        clip: Optional[Tuple[float, float]] = None,
    ) -> "RewardNormalizer":
        """Fit statistics from ``data["rewards"]``.

        Args:
            data:            Dataset dictionary containing a ``"rewards"``
                             key.  The value must be a NumPy array of shape
                             ``(N,)`` or ``(N, 1)``.
            normalizer_cls:  Which base normalizer class to fit.  Defaults to
                             :class:`GaussianNormalizer`.  Must be a subclass
                             of :class:`Normalizer` with a compatible
                             :meth:`fit` class method.
            clip:            Optional ``(lo, hi)`` clip bounds in the
                             *normalised* space.

        Returns:
            Fitted :class:`RewardNormalizer`.
        """
        rewards = _extract_array(data, "rewards")
        # Ensure 2-D for the base normalizer (N, 1).
        rew_2d = rewards.reshape(-1, 1)
        base = normalizer_cls.fit({"rewards": rew_2d}, key="rewards")
        return cls(base_normalizer=base, clip=clip)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        """Return serialisable state for checkpointing."""
        return {
            "base_normalizer_cls": self._norm.__class__.__name__,
            "base_stats": self._norm.state_dict(),
            "clip": list(self._clip) if self._clip is not None else None,
        }

    def __repr__(self) -> str:
        return (
            f"RewardNormalizer("
            f"base={self._norm!r}, "
            f"clip={self._clip})"
        )


# ---------------------------------------------------------------------------
# Dataset-level factory helpers
# ---------------------------------------------------------------------------


def fit_normalizers(
    data: DataDict,
    normalizer_cls: type = LimitsNormalizer,
    reward_normalizer_cls: type = GaussianNormalizer,
    reward_clip: Optional[Tuple[float, float]] = None,
    keys: Sequence[str] = SUPPORTED_KEYS,
) -> Dict[str, Union[Normalizer, RewardNormalizer]]:
    """Fit a normalizer for every key present in *data*.

    Convenience factory that inspects which of ``"observations"``,
    ``"actions"``, and ``"rewards"`` exist in *data* and fits the appropriate
    normalizer.

    Args:
        data:                  Dataset dictionary.
        normalizer_cls:        Class used for ``"observations"`` and
                               ``"actions"`` (default:
                               :class:`LimitsNormalizer`).
        reward_normalizer_cls: Base normalizer class passed to
                               :class:`RewardNormalizer` for the
                               ``"rewards"`` key (default:
                               :class:`GaussianNormalizer`).
        reward_clip:           Optional clip bounds in normalised reward space.
        keys:                  Subset of ``SUPPORTED_KEYS`` to process.
                               Defaults to all three.

    Returns:
        Dictionary mapping each present key to its fitted normalizer.
        ``"rewards"`` maps to a :class:`RewardNormalizer`; other keys map to
        an instance of *normalizer_cls*.

    Example::

        normalizers = fit_normalizers(data)
        obs_n  = normalizers["observations"].normalize(obs_batch)
        rew_n  = normalizers["rewards"].normalize(rew_batch)
    """
    normalizers: Dict[str, Union[Normalizer, RewardNormalizer]] = {}

    for key in keys:
        if key not in data:
            continue
        if key == "rewards":
            normalizers[key] = RewardNormalizer.fit(
                data,
                normalizer_cls=reward_normalizer_cls,
                clip=reward_clip,
            )
        else:
            normalizers[key] = normalizer_cls.fit(data, key=key)

    return normalizers


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_array(data: DataDict, key: str) -> Array:
    """Extract and validate an array from *data[key]*.

    Args:
        data: Dataset dictionary.
        key:  Key to extract.

    Returns:
        2-D float64 array of shape ``(N, d)`` (or ``(N, 1)`` for scalars).

    Raises:
        KeyError:   If *key* is absent from *data*.
        ValueError: If the resulting array has zero samples.
    """
    if key not in data:
        raise KeyError(
            f"Key '{key}' not found in data dict. "
            f"Available keys: {list(data.keys())}"
        )
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(
            f"Expected a 1-D or 2-D array for key='{key}'; "
            f"got shape {arr.shape}."
        )
    if arr.shape[0] == 0:
        raise ValueError(
            f"Cannot fit normalizer on an empty array (key='{key}')."
        )
    return arr
