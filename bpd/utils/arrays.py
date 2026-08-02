"""
Array utility functions for numpy/torch conversions and device management.

Modeled after diffuser/utils/arrays.py from the Diffuser codebase
(Janner et al., 2022, NeurIPS).
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Named tuple for a standard RL transition batch
# ---------------------------------------------------------------------------

DataBatch = namedtuple("DataBatch", ["state", "action", "reward", "next_state", "done"])


# ---------------------------------------------------------------------------
# Device / tensor movement helpers
# ---------------------------------------------------------------------------


def batch_to_device(
    batch: Union[DataBatch, Dict[str, torch.Tensor]],
    device: Union[str, torch.device],
) -> Union[DataBatch, Dict[str, torch.Tensor]]:
    """Move every tensor in *batch* to *device*.

    Accepts either a :class:`DataBatch` namedtuple or a plain ``dict`` whose
    values are :class:`torch.Tensor` objects.  The return type mirrors the
    input type.

    Args:
        batch: A :class:`DataBatch` namedtuple or a ``dict`` of tensors.
        device: Target device (e.g. ``"cuda"``, ``"cpu"``, or a
            :class:`torch.device` instance).

    Returns:
        The same structure with every tensor moved to *device*.
    """
    device = torch.device(device)

    if isinstance(batch, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    if isinstance(batch, DataBatch):
        fields = {
            field: getattr(batch, field).to(device)
            if isinstance(getattr(batch, field), torch.Tensor)
            else getattr(batch, field)
            for field in batch._fields
        }
        return DataBatch(**fields)

    raise TypeError(
        f"batch_to_device expects a DataBatch namedtuple or a dict, "
        f"got {type(batch).__name__!r}."
    )


# ---------------------------------------------------------------------------
# Numpy <-> Torch conversions
# ---------------------------------------------------------------------------


def to_np(x: torch.Tensor) -> np.ndarray:
    """Convert a :class:`torch.Tensor` to a NumPy ``ndarray``.

    The tensor is detached from the computational graph and moved to CPU
    before conversion, so gradients are not preserved.

    Args:
        x: Input tensor (any device, any dtype).

    Returns:
        A ``numpy.ndarray`` with the same data.
    """
    return x.detach().cpu().numpy()


def to_torch(
    x: np.ndarray,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Convert a NumPy ``ndarray`` to a :class:`torch.Tensor`.

    Args:
        x: Input array.
        dtype: Desired PyTorch dtype.  If ``None`` the dtype is inferred
            from *x* (``numpy.float64`` maps to ``torch.float32`` to match
            common deep-learning conventions).
        device: Target device.  If ``None`` the tensor is left on CPU.

    Returns:
        A :class:`torch.Tensor` on *device* with the requested *dtype*.
    """
    if dtype is None:
        # Promote float64 → float32 for GPU/MPS efficiency; preserve others.
        if x.dtype == np.float64:
            dtype = torch.float32
        # All other dtypes (int32, int64, bool, …) are inferred automatically.

    tensor = torch.from_numpy(x)

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)

    if device is not None:
        tensor = tensor.to(device=torch.device(device))

    return tensor


# ---------------------------------------------------------------------------
# Dict helpers
# ---------------------------------------------------------------------------


def apply_dict(
    fn: Callable[[Any], Any],
    d: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply *fn* to every value in the dictionary *d*.

    Args:
        fn: A callable that maps a single value to a transformed value.
        d:  The source dictionary.

    Returns:
        A new ``dict`` with the same keys and transformed values.
    """
    return {k: fn(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def atleast_2d(x: np.ndarray) -> np.ndarray:
    """Ensure that array *x* has at least two dimensions.

    Scalars are reshaped to ``(1, 1)`` and 1-D arrays to ``(1, N)``,
    consistent with :func:`numpy.atleast_2d`.

    Args:
        x: Input array of any shape.

    Returns:
        An array with ``x.ndim >= 2``.
    """
    return np.atleast_2d(x)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize(
    x: np.ndarray,
    mean: Union[np.ndarray, float],
    std: Union[np.ndarray, float],
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize *x* to zero mean and unit variance.

    .. math::
        x_{\\text{norm}} = \\frac{x - \\mu}{\\sigma + \\epsilon}

    Args:
        x:    Input array.
        mean: Per-feature mean (broadcastable with *x*).
        std:  Per-feature standard deviation (broadcastable with *x*).
        eps:  Small constant added to *std* for numerical stability.

    Returns:
        Normalized array with the same shape as *x*.
    """
    return (x - mean) / (std + eps)


def unnormalize(
    x: np.ndarray,
    mean: Union[np.ndarray, float],
    std: Union[np.ndarray, float],
) -> np.ndarray:
    """Reverse a previous :func:`normalize` call.

    .. math::
        x = x_{\\text{norm}} \\cdot \\sigma + \\mu

    Args:
        x:    Normalized input array.
        mean: Per-feature mean used during normalization (broadcastable with *x*).
        std:  Per-feature standard deviation used during normalization
              (broadcastable with *x*).

    Returns:
        Unnormalized array with the same shape as *x*.
    """
    return x * std + mean
