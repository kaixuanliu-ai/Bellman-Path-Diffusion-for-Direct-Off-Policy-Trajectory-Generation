"""Performance configuration for GPU + multi-threaded CPU execution.

Centralizes the throughput knobs so training and evaluation saturate the GPU
without the CPU becoming a bottleneck:

* multi-threaded CPU intra-op / inter-op pools (``torch.set_num_threads``),
* cuDNN autotuner (``cudnn.benchmark``) for fixed-shape trajectory tensors,
* TF32 matmul/conv on Ampere+ GPUs for higher throughput at negligible OPE cost.

The BPD training loop keeps the whole transition dataset resident on the GPU
(see ``BellmanPathDiffusionTrainer``), so per-step host->device copies are
avoided and the GPU never waits on a CPU DataLoader.  The CPU threads configured
here are used for the remaining host-side work (indexing, replay bookkeeping).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolve a device string, defaulting to CUDA when available.

    Args:
        device: Explicit device string (e.g. ``"cuda"``, ``"cuda:0"``,
                ``"cpu"``).  If ``None``, uses ``"cuda"`` when available else
                ``"cpu"``.

    Returns:
        A ``torch.device``.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_performance(
    device: torch.device,
    num_threads: Optional[int] = None,
    allow_tf32: bool = True,
    cudnn_benchmark: bool = True,
) -> None:
    """Configure CPU threading and GPU throughput backends.

    Args:
        device:          Compute device the run will use.
        num_threads:     Number of CPU threads for intra-op parallelism.  If
                         ``None``, uses ``os.cpu_count()`` (all cores).
        allow_tf32:      Enable TF32 matmul/conv on CUDA (Ampere+); ignored on
                         CPU and on GPUs without TF32 (e.g. Titan RTX falls back
                         to the standard path safely).
        cudnn_benchmark: Enable the cuDNN autotuner.  Beneficial because BPD
                         trajectory tensors have fixed (B, h, d_tok) shapes per
                         stage.
    """
    threads = int(num_threads) if num_threads else (os.cpu_count() or 1)
    threads = max(1, threads)
    torch.set_num_threads(threads)
    try:
        # Inter-op pool: keep modest so it does not oversubscribe with intra-op.
        torch.set_num_interop_threads(max(1, min(4, threads)))
    except RuntimeError:
        # set_num_interop_threads can only be called once per process.
        pass

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        try:
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        except Exception:  # pragma: no cover - older torch
            pass
        name = torch.cuda.get_device_name(device)
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        logger.info(
            "perf: device=%s (%s, %.1f GB), cpu_threads=%d, tf32=%s, "
            "cudnn.benchmark=%s",
            device, name, total_gb, threads, allow_tf32, cudnn_benchmark,
        )
    else:
        logger.info("perf: device=cpu, cpu_threads=%d", threads)


def amp_enabled(device: torch.device, requested: bool) -> bool:
    """Return whether autocast mixed precision should be active.

    Mixed precision (and the CUDA GradScaler) are only used on CUDA; on CPU the
    training step runs in float32 for numerical stability of the score targets.

    Args:
        device:    Compute device.
        requested: User request for AMP.

    Returns:
        ``True`` iff AMP should be enabled.
    """
    return bool(requested) and device.type == "cuda"
