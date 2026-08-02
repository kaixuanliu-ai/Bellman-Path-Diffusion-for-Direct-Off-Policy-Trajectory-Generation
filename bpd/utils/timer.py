"""
Wall-clock timer utilities modeled after the timer in Janner et al. 2022 (Diffuser).

Provides:
    Timer     -- a simple start/stop wall-clock timer with context-manager support.
    TimeIt    -- a context manager that prints elapsed time with a label.
    step_timer -- a module-level Timer instance for accumulating per-step timing.
"""

import time


class Timer:
    """A simple wall-clock timer that supports start/stop and context-manager usage.

    Typical usage::

        t = Timer()
        t.start()
        do_work()
        elapsed = t.stop()   # seconds as float

    Or as a context manager::

        with Timer() as t:
            do_work()
        print(t.elapsed)
    """

    def __init__(self) -> None:
        self._start_time: float | None = None
        self._elapsed: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "Timer":
        """Start (or resume) the timer. Returns self for chaining."""
        if self._start_time is not None:
            raise RuntimeError(
                "Timer is already running. Call stop() before calling start() again."
            )
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return the elapsed wall-clock time in seconds.

        Accumulates across multiple start/stop cycles so the timer can be
        used as a lap counter.
        """
        if self._start_time is None:
            raise RuntimeError(
                "Timer is not running. Call start() before calling stop()."
            )
        self._elapsed += time.perf_counter() - self._start_time
        self._start_time = None
        return self._elapsed

    def reset(self) -> None:
        """Reset accumulated elapsed time and stop the timer if running."""
        self._start_time = None
        self._elapsed = 0.0

    @property
    def elapsed(self) -> float:
        """Total accumulated elapsed time in seconds.

        If the timer is currently running the time since the last start()
        call is included in the returned value.
        """
        if self._start_time is not None:
            return self._elapsed + (time.perf_counter() - self._start_time)
        return self._elapsed

    @property
    def is_running(self) -> bool:
        """True while the timer is between a start() and stop() call."""
        return self._start_time is not None

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False  # do not suppress exceptions

    def __repr__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return f"Timer(elapsed={self.elapsed:.6f}s, status={status})"


class TimeIt:
    """Context manager that measures and prints wall-clock time for a block.

    Usage::

        with TimeIt("forward pass"):
            logits = model(batch)
        # Prints: forward pass: 0.0123 s

    Parameters
    ----------
    label:
        A human-readable description printed before the elapsed time.
    fmt:
        Optional printf-style format string for the elapsed seconds.
        Defaults to ``".4f"`` (four decimal places).
    verbose:
        Set to False to suppress printing (useful for conditional logging).
    """

    def __init__(
        self,
        label: str = "",
        fmt: str = ".4f",
        verbose: bool = True,
    ) -> None:
        self.label = label
        self.fmt = fmt
        self.verbose = verbose
        self._timer = Timer()
        self.elapsed: float = 0.0

    def __enter__(self) -> "TimeIt":
        self._timer.reset()
        self._timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.elapsed = self._timer.stop()
        if self.verbose:
            label = f"{self.label}: " if self.label else ""
            print(f"{label}{self.elapsed:{self.fmt}} s")
        return False  # do not suppress exceptions

    def __repr__(self) -> str:
        return f"TimeIt(label={self.label!r}, elapsed={self.elapsed:.6f}s)"


class _StepTimer:
    """A module-level timer designed to accumulate per-step timing statistics.

    Typical usage inside a training loop::

        step_timer.start()
        loss = train_step(batch)
        step_time = step_timer.step()   # seconds for this step

        if step % log_every == 0:
            print(f"avg step time: {step_timer.average:.4f} s")
            step_timer.reset()

    Alternatively used as a context manager::

        with step_timer:
            loss = train_step(batch)
    """

    def __init__(self) -> None:
        self._timer = Timer()
        self._total: float = 0.0
        self._count: int = 0
        self._last: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mark the beginning of a step."""
        if self._timer.is_running:
            self._timer.stop()
            self._timer.reset()
        self._timer.start()

    def step(self) -> float:
        """Mark the end of a step, accumulate, and return the step duration."""
        if not self._timer.is_running:
            raise RuntimeError(
                "step_timer is not running. Call start() before step()."
            )
        self._last = self._timer.stop()
        self._total += self._last
        self._count += 1
        self._timer.reset()
        return self._last

    def reset(self) -> None:
        """Reset all accumulated statistics and stop the internal timer."""
        self._timer.reset()
        self._total = 0.0
        self._count = 0
        self._last = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total(self) -> float:
        """Total accumulated time across all completed steps (seconds)."""
        return self._total

    @property
    def count(self) -> int:
        """Number of completed steps."""
        return self._count

    @property
    def last(self) -> float:
        """Duration of the most recently completed step (seconds)."""
        return self._last

    @property
    def average(self) -> float:
        """Mean step duration in seconds. Returns 0.0 if no steps recorded."""
        if self._count == 0:
            return 0.0
        return self._total / self._count

    # ------------------------------------------------------------------
    # Context-manager support (one step per with-block)
    # ------------------------------------------------------------------

    def __enter__(self) -> "_StepTimer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.step()
        return False

    def __repr__(self) -> str:
        return (
            f"_StepTimer("
            f"count={self._count}, "
            f"total={self._total:.6f}s, "
            f"average={self.average:.6f}s)"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: A global _StepTimer instance ready to use across modules without
#: needing to instantiate one explicitly.
step_timer: _StepTimer = _StepTimer()
