"""
Serialization utilities for saving and loading model checkpoints and configs.

Patterned after diffuser/utils/serialization.py (Janner et al.) and IQL
checkpointing conventions.  All public functions raise descriptive exceptions
on failure rather than silently returning None.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    state_dict: Dict[str, Any],
    path: PathLike,
    filename: str = "checkpoint.pt",
) -> Path:
    """Save *state_dict* as a PyTorch checkpoint file.

    Parameters
    ----------
    state_dict:
        Arbitrary dict that ``torch.save`` can serialise.  Typically contains
        ``{"model": model.state_dict(), "optimizer": opt.state_dict(),
        "step": int, ...}``.
    path:
        Directory in which the file will be written.  Created recursively if
        it does not already exist.
    filename:
        Base name of the output file (e.g. ``"checkpoint_step_1000.pt"``).

    Returns
    -------
    Path
        Absolute path of the saved file.

    Raises
    ------
    TypeError
        If *state_dict* is not a dict.
    OSError
        If the directory cannot be created or the file cannot be written.
    """
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"state_dict must be a dict, got {type(state_dict).__name__}"
        )

    save_dir = Path(path)
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Cannot create checkpoint directory '{save_dir}': {exc}"
        ) from exc

    dest = save_dir / filename
    try:
        torch.save(state_dict, dest)
    except Exception as exc:
        raise OSError(
            f"Failed to write checkpoint to '{dest}': {exc}"
        ) from exc

    logger.info("Saved checkpoint -> %s", dest)
    return dest.resolve()


def load_checkpoint(
    path: PathLike,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Load a PyTorch checkpoint from *path*.

    Parameters
    ----------
    path:
        Full path to a ``.pt`` / ``.pth`` file **or** a directory, in which
        case the most recent checkpoint is loaded automatically (via
        :func:`get_latest_checkpoint`).
    device:
        Target device for tensors.  Defaults to ``"cpu"`` if not specified,
        which avoids GPU memory errors when checkpoints are transferred across
        machines.

    Returns
    -------
    dict
        The state dict that was previously passed to :func:`save_checkpoint`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist or no ``.pt`` file is found in the directory.
    RuntimeError
        If the checkpoint cannot be deserialised.
    """
    ckpt_path = Path(path)

    if ckpt_path.is_dir():
        ckpt_path = get_latest_checkpoint(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: '{ckpt_path}'")

    map_location: Union[str, torch.device] = device if device is not None else "cpu"

    try:
        state_dict = torch.load(ckpt_path, map_location=map_location)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint from '{ckpt_path}': {exc}"
        ) from exc

    logger.info("Loaded checkpoint <- %s  (device=%s)", ckpt_path, map_location)
    return state_dict


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def save_config(config: Dict[str, Any], path: PathLike) -> Path:
    """Serialise *config* to a pretty-printed JSON file.

    Parameters
    ----------
    config:
        A JSON-serialisable dictionary (e.g. hyperparameters, architecture
        settings, dataset paths).
    path:
        Destination file path **or** directory.  If *path* is a directory the
        config is saved as ``config.json`` inside it.

    Returns
    -------
    Path
        Absolute path of the written JSON file.

    Raises
    ------
    TypeError
        If *config* is not a dict or contains non-serialisable values.
    OSError
        If the file cannot be written.
    """
    if not isinstance(config, dict):
        raise TypeError(
            f"config must be a dict, got {type(config).__name__}"
        )

    dest = Path(path)
    if dest.is_dir() or not dest.suffix:
        dest = dest / "config.json"

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Cannot create config directory '{dest.parent}': {exc}"
        ) from exc

    try:
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"config contains non-serialisable values: {exc}"
        ) from exc
    except OSError as exc:
        raise OSError(f"Failed to write config to '{dest}': {exc}") from exc

    logger.info("Saved config -> %s", dest)
    return dest.resolve()


def load_config(path: PathLike) -> Dict[str, Any]:
    """Load a JSON config file produced by :func:`save_config`.

    Parameters
    ----------
    path:
        Full path to the JSON file.

    Returns
    -------
    dict
        Deserialised configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is not valid JSON or does not decode to a dict.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Config file not found: '{src}'")

    try:
        with src.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"'{src}' is not valid JSON: {exc}"
        ) from exc
    except OSError as exc:
        raise OSError(f"Cannot read config from '{src}': {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a JSON object at the top level, got {type(config).__name__}"
        )

    logger.info("Loaded config <- %s", src)
    return config


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"(\d+)")


def _checkpoint_sort_key(p: Path) -> int:
    """Return the last integer found in the filename, or 0 if none."""
    numbers = _STEP_RE.findall(p.stem)
    return int(numbers[-1]) if numbers else 0


def get_latest_checkpoint(checkpoint_dir: PathLike) -> Path:
    """Return the path of the most recent ``.pt`` file in *checkpoint_dir*.

    "Most recent" is determined first by the largest integer embedded in the
    filename (e.g. ``checkpoint_step_5000.pt`` > ``checkpoint_step_1000.pt``),
    and then by modification time as a tiebreaker.

    Parameters
    ----------
    checkpoint_dir:
        Directory to search (non-recursive).

    Returns
    -------
    Path
        Absolute path of the latest checkpoint.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_dir* does not exist or contains no ``.pt`` files.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: '{ckpt_dir}'"
        )

    candidates = sorted(
        ckpt_dir.glob("*.pt"),
        key=lambda p: (_checkpoint_sort_key(p), p.stat().st_mtime),
    )

    # Also pick up .pth files used by some frameworks.
    candidates += sorted(
        ckpt_dir.glob("*.pth"),
        key=lambda p: (_checkpoint_sort_key(p), p.stat().st_mtime),
    )

    if not candidates:
        raise FileNotFoundError(
            f"No .pt/.pth checkpoints found in '{ckpt_dir}'"
        )

    latest = max(
        candidates,
        key=lambda p: (_checkpoint_sort_key(p), p.stat().st_mtime),
    )
    logger.debug("Latest checkpoint in '%s': %s", ckpt_dir, latest.name)
    return latest.resolve()


# ---------------------------------------------------------------------------
# Serializable mixin
# ---------------------------------------------------------------------------

class Serializable(ABC):
    """Mixin that provides JSON-compatible serialisation for config dataclasses.

    Subclasses should either:

    * Be standard Python dataclasses (``@dataclass``), in which case
      :meth:`to_dict` / :meth:`from_dict` work automatically via
      ``__dataclass_fields__``.
    * Override :meth:`to_dict` and :meth:`from_dict` for custom logic.

    Example
    -------
    ::

        @dataclass
        class TrainingConfig(Serializable):
            lr: float = 3e-4
            batch_size: int = 256
            horizon: int = 32

        cfg = TrainingConfig(lr=1e-3)
        d   = cfg.to_dict()            # {"lr": 1e-3, "batch_size": 256, ...}
        cfg2 = TrainingConfig.from_dict(d)
    """

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict representation.

        The default implementation introspects ``__dataclass_fields__`` if the
        subclass is a dataclass, and falls back to ``__dict__`` otherwise.
        Values that are themselves :class:`Serializable` are recursively
        converted.

        Returns
        -------
        dict
        """
        if hasattr(self, "__dataclass_fields__"):
            fields: Dict[str, Any] = {}
            for name in self.__dataclass_fields__:  # type: ignore[attr-defined]
                value = getattr(self, name)
                fields[name] = (
                    value.to_dict() if isinstance(value, Serializable) else value
                )
            return fields

        # Generic fallback: shallow copy of instance __dict__
        return {
            k: (v.to_dict() if isinstance(v, Serializable) else v)
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Serializable":
        """Reconstruct an instance from a plain dict.

        For dataclass subclasses this calls ``cls(**data)``.  Override for
        richer deserialisation (e.g. nested objects, version migrations).

        Parameters
        ----------
        data:
            Dict as returned by :meth:`to_dict`.

        Returns
        -------
        Serializable
            A new instance of the calling class.

        Raises
        ------
        TypeError
            If *data* is not a dict or contains unexpected keys.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"from_dict expects a dict, got {type(data).__name__}"
            )
        try:
            return cls(**data)
        except TypeError as exc:
            raise TypeError(
                f"Cannot construct {cls.__name__} from provided dict: {exc}"
            ) from exc

    def save(self, path: PathLike) -> Path:
        """Convenience wrapper: serialise and write to *path* via
        :func:`save_config`.

        Parameters
        ----------
        path:
            File or directory destination (see :func:`save_config`).

        Returns
        -------
        Path
            Absolute path of the written file.
        """
        return save_config(self.to_dict(), path)

    @classmethod
    def load(cls, path: PathLike) -> "Serializable":
        """Convenience wrapper: read JSON from *path* and call
        :meth:`from_dict`.

        Parameters
        ----------
        path:
            JSON file path (see :func:`load_config`).

        Returns
        -------
        Serializable
            A new instance of the calling class.
        """
        data = load_config(path)
        return cls.from_dict(data)
