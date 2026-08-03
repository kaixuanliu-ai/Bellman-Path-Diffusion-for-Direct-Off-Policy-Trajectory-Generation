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
import time
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


def maybe_compile(module, enabled: bool = True, dynamic: bool = True):
    """Optionally wrap *module* with ``torch.compile``.

    The BPD score network is a stack of small GEMMs (model_dim is typically a
    few hundred), so a large share of step time is kernel-launch overhead rather
    than math.  ``torch.compile`` fuses those ops and typically recovers a solid
    fraction of it.

    ``dynamic=True`` matters here: the horizon ``h`` changes the sequence
    length, and a static compile would trigger a fresh recompilation for every
    horizon (H recompiles, often costing more than it saves).

    Falls back to the eager module if compilation is unavailable or fails, so
    enabling this can never break a run.

    Args:
        module:  The ``nn.Module`` to compile.
        enabled: Set False to keep the eager module.
        dynamic: Compile with dynamic shapes (recommended for variable h).

    Returns:
        The compiled module, or the original on failure / when disabled.
    """
    if not enabled or not hasattr(torch, "compile"):
        return module
    try:
        compiled = torch.compile(module, dynamic=dynamic)
        logger.info("perf: torch.compile enabled (dynamic=%s)", dynamic)
        return compiled
    except Exception as exc:  # pragma: no cover - backend/toolchain dependent
        logger.warning("perf: torch.compile unavailable (%s); using eager", exc)
        return module


class ThroughputMeter:
    """Track training throughput: samples/s, tokens/s and effective TFLOPS.

    Utilization percentages from ``nvidia-smi`` only report that *some* kernel
    was resident; they stay near 100% even when the GPU is starved by tiny
    launches or is waiting on the host.  Throughput is the metric that actually
    exposes those stalls, so the trainer reports it directly.

    Args:
        n_params:    Trainable parameter count of the network (for the
                     6*N*tokens forward+backward FLOP estimate).
        device:      Compute device (used to synchronize before timing).
        window:      Number of steps averaged per report.
    """

    def __init__(self, n_params: int, device: torch.device, window: int = 100):
        self.n_params = int(n_params)
        self.device = torch.device(device)
        self.window = max(1, int(window))
        self._samples = 0
        self._tokens = 0
        self._steps = 0
        self._t0 = time.perf_counter()

    def update(self, batch_size: int, seq_len: int) -> Optional[dict]:
        """Record one step; returns a stats dict once per window, else None."""
        self._samples += int(batch_size)
        self._tokens += int(batch_size) * int(seq_len)
        self._steps += 1
        if self._steps < self.window:
            return None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        dt = max(1e-9, time.perf_counter() - self._t0)
        stats = {
            "samples_per_s": self._samples / dt,
            "tokens_per_s": self._tokens / dt,
            "tflops": 6.0 * self.n_params * self._tokens / dt / 1e12,
            "ms_per_step": 1000.0 * dt / self._steps,
        }
        if self.device.type == "cuda":
            stats["vram_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
            total = torch.cuda.get_device_properties(self.device).total_memory
            stats["vram_pct"] = 100.0 * torch.cuda.max_memory_allocated(self.device) / total
        self._samples = self._tokens = self._steps = 0
        self._t0 = time.perf_counter()
        return stats


def format_throughput(stats: dict) -> str:
    """Render :class:`ThroughputMeter` stats as a compact log line."""
    parts = [
        f"{stats['samples_per_s']:,.0f} samples/s",
        f"{stats['tokens_per_s']/1e3:,.0f}k tokens/s",
        f"{stats['tflops']:.1f} TFLOPS",
        f"{stats['ms_per_step']:.1f} ms/step",
    ]
    if "vram_gb" in stats:
        parts.append(f"VRAM {stats['vram_gb']:.1f}GB ({stats['vram_pct']:.0f}%)")
    return "  ".join(parts)


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
