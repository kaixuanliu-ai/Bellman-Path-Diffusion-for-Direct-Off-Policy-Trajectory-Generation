"""
Lightweight logging utility for BPD experiments.

Supports console output, timestamped file logging, and optional W&B integration.
Modeled after logger patterns in IQL, Decision Diffuser, and TD-MPC codebases.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Module-level registry so callers can share a single Logger instance.
# ---------------------------------------------------------------------------
_GLOBAL_LOGGER: Optional["Logger"] = None


def get_logger() -> "Logger":
    """Return the module-level Logger instance.

    Raises RuntimeError if no Logger has been instantiated yet.
    """
    if _GLOBAL_LOGGER is None:
        raise RuntimeError(
            "No Logger has been created yet. "
            "Instantiate Logger(...) before calling get_logger()."
        )
    return _GLOBAL_LOGGER


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_file_handler(log_dir: Path, run_name: Optional[str]) -> logging.FileHandler:
    """Create a timestamped FileHandler under log_dir."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = run_name if run_name else "run"
    filename = f"{stem}_{timestamp}.log"
    log_path = log_dir / filename
    handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    return handler


def _format_metrics(metrics: Dict[str, Any], prefix: str, step: int) -> str:
    """Format a metrics dict into a readable one-line string."""
    tag = f"[{prefix}] " if prefix else ""
    kv_pairs = "  ".join(f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}"
                          for k, v in sorted(metrics.items()))
    return f"{tag}step={step}  {kv_pairs}"


# ---------------------------------------------------------------------------
# Main Logger class
# ---------------------------------------------------------------------------

class Logger:
    """Unified logger for BPD training runs.

    Parameters
    ----------
    log_dir : str or Path
        Directory where log files and config snapshots are stored.
        Created automatically if it does not exist.
    use_wandb : bool
        If True, attempt to import wandb and log metrics there as well.
    project_name : str
        W&B project name (ignored when use_wandb=False).
    run_name : str, optional
        Human-readable label for this run.  Used as both the W&B run name
        and as a prefix in the log filename.  Defaults to a timestamp.
    """

    def __init__(
        self,
        log_dir: str,
        use_wandb: bool = False,
        project_name: str = "bpd",
        run_name: Optional[str] = None,
    ) -> None:
        global _GLOBAL_LOGGER

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        self.project_name = project_name
        self.run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")

        # --- Python logger setup -------------------------------------------
        logger_name = f"bpd.{self.run_name}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # avoid double-printing to root logger

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        if not any(isinstance(h, logging.StreamHandler) for h in self._logger.handlers):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        # File handler
        file_handler = _build_file_handler(self.log_dir, self.run_name)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        # Store the path for external reference
        self.log_file: Path = Path(file_handler.baseFilename)

        # --- Optional W&B init ---------------------------------------------
        self._wandb = None
        if use_wandb:
            try:
                import wandb  # type: ignore
                self._wandb = wandb
                if wandb.run is None:
                    wandb.init(project=project_name, name=self.run_name)
                self._logger.info("W&B initialized: project=%s  run=%s", project_name, self.run_name)
            except ImportError:
                self._logger.warning(
                    "use_wandb=True but 'wandb' is not installed. "
                    "Falling back to file/console logging only."
                )
                self.use_wandb = False

        self._logger.info(
            "Logger started | log_dir=%s | run=%s | wandb=%s",
            self.log_dir,
            self.run_name,
            self.use_wandb,
        )

        # Register as the module-level singleton
        _GLOBAL_LOGGER = self

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log a dictionary of scalar metrics.

        Parameters
        ----------
        metrics : dict
            Mapping of metric name to numeric value.
        step : int
            Global training step or environment step.
        prefix : str
            Optional namespace prefix (e.g. ``'train'`` or ``'eval'``).
            Prepended to each key when logging to W&B as ``prefix/key``.
        """
        if not metrics:
            return

        # Console / file
        msg = _format_metrics(metrics, prefix, step)
        self._logger.info(msg)

        # W&B
        if self.use_wandb and self._wandb is not None:
            sep = "/" if prefix else ""
            wandb_dict = {f"{prefix}{sep}{k}": v for k, v in metrics.items()}
            self._wandb.log(wandb_dict, step=step)

    def log_string(self, key: str, value: str, step: int) -> None:
        """Log a single string-valued entry.

        Parameters
        ----------
        key : str
            Identifier for the logged value.
        value : str
            String payload.
        step : int
            Global training step or environment step.
        """
        self._logger.info("step=%d  %s: %s", step, key, value)

        if self.use_wandb and self._wandb is not None:
            # W&B does not natively support arbitrary string metrics; log as
            # a text artifact instead.
            self._wandb.log({key: self._wandb.Html(f"<pre>{value}</pre>")}, step=step)

    def save_config(self, config: Dict[str, Any]) -> Path:
        """Serialize *config* to ``<log_dir>/config.json``.

        Parameters
        ----------
        config : dict
            Experiment hyper-parameters / settings.

        Returns
        -------
        Path
            Absolute path to the written JSON file.
        """
        config_path = self.log_dir / "config.json"

        # Annotate with run metadata
        enriched = {
            "_run_name": self.run_name,
            "_saved_at": datetime.now().isoformat(),
            **config,
        }

        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(enriched, fh, indent=2, default=str)

        self._logger.info("Config saved to %s", config_path)

        if self.use_wandb and self._wandb is not None:
            # W&B accepts plain dicts directly
            self._wandb.config.update(config, allow_val_change=True)

        return config_path

    # -----------------------------------------------------------------------
    # Convenience pass-throughs for raw logging levels
    # -----------------------------------------------------------------------

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(msg, *args, **kwargs)

    def close(self) -> None:
        """Flush and close all handlers.  Finish the W&B run if active."""
        for handler in list(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)

        if self.use_wandb and self._wandb is not None and self._wandb.run is not None:
            self._wandb.finish()

    # -----------------------------------------------------------------------
    # Context manager support
    # -----------------------------------------------------------------------

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Logger(run_name={self.run_name!r}, "
            f"log_dir={str(self.log_dir)!r}, "
            f"use_wandb={self.use_wandb})"
        )
