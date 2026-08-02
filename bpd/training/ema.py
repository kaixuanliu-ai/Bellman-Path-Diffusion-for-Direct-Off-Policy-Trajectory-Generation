"""
Exponential Moving Average (EMA) of model parameters for Bellman Path Diffusion.

Implements the EMA target network update rule from Appendix D of the paper:

    θ̄ ← ζ θ̄ + (1-ζ) θ   (after each gradient update step)

The shadow model s_{θ̄} is used as the frozen teacher network in the
continuation branch of BPD, providing stable targets during training.

Design notes:
  - The shadow model is a deep copy of the online model at construction time.
    Gradients on the shadow model are always disabled.
  - Updates are skipped for the first ``update_after_step`` steps to allow
    the online model to warm up before EMA tracking begins.
  - After the warm-up period, parameter updates are applied every
    ``update_every`` steps to amortise the copy overhead.
  - All public methods acquire a reentrant-safe threading lock, making the
    class safe to call from multiple threads (e.g. a DataLoader worker
    calling ``get_model()`` while the main thread calls ``update()``).

References:
  - BPD paper, Appendix D (EMA targets).
  - diffuser/utils/training.py EMAModel (Janner et al.).
  - denoising_diffusion_pytorch (Lucidrains), EMA class.
  - Polyak & Juditsky 1992; Izmailov et al. 2018 (SWA/EMA).
"""

from __future__ import annotations

import copy
import threading
from typing import Any

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of the parameters of an ``nn.Module``.

    Maintains a shadow copy of the model whose parameters are updated via:

        θ̄ ← ζ θ̄ + (1-ζ) θ

    after each qualifying training step.  The shadow model is used as the
    frozen teacher (s_{θ̄}) in the BPD continuation branch.

    Args:
        model:             The online model whose parameters will be tracked.
                           A deep copy is taken immediately; the original is
                           not modified.
        decay:             EMA decay coefficient ζ ∈ (0, 1).  Higher values
                           give a slower-moving average.  Default: 0.995.
        update_every:      Apply the EMA update only once every this many calls
                           to ``update()``.  Steps in between are counted but
                           the shadow weights are not changed.  Default: 10.
        update_after_step: Do not update until this many steps have been taken.
                           Allows the online model to leave the random
                           initialisation before EMA begins tracking.
                           Default: 100.

    Thread safety:
        A ``threading.RLock`` guards all methods that read or write shared
        state (``update``, ``copy_params_from_model_to_ema``, ``get_model``,
        ``state_dict``, ``load_state_dict``).  The lock is reentrant so that
        subclasses or callers holding the lock may safely call these methods
        recursively without deadlocking.

    Example::

        ema = EMA(model, decay=0.995, update_every=10, update_after_step=100)

        for step, batch in enumerate(loader):
            loss = compute_loss(model, batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            ema.update(model, step=step)

        shadow = ema.get_model()
        with torch.no_grad():
            predictions = shadow(inputs)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.995,
        update_every: int = 10,
        update_after_step: int = 100,
    ) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in the open interval (0, 1); got {decay}")
        if update_every < 1:
            raise ValueError(f"update_every must be >= 1; got {update_every}")
        if update_after_step < 0:
            raise ValueError(f"update_after_step must be >= 0; got {update_after_step}")

        self.decay = decay
        self.update_every = update_every
        self.update_after_step = update_after_step

        # Internal step counter — incremented on every call to update().
        self._step: int = 0

        # Flag: has the first qualifying EMA update been performed?
        # On the very first qualifying step we hard-copy instead of blending
        # so the shadow immediately reflects current online weights rather than
        # blending against the initial random clone.
        self._initialized: bool = False

        # Reentrant lock for thread safety.
        self._lock: threading.RLock = threading.RLock()

        # Shadow model: deep copy, gradients permanently disabled.
        self._shadow: nn.Module = copy.deepcopy(model)
        self._shadow.eval()
        for param in self._shadow.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, model: nn.Module, step: int) -> None:
        """Update the EMA shadow weights from the online model.

        The update is applied only when:
          1. ``step`` (or the internal counter) has passed ``update_after_step``.
          2. The internal counter is a multiple of ``update_every``.

        Args:
            model: The online model providing current parameters θ.
            step:  External training step counter.  Used to decide whether the
                   warm-up period has elapsed.  If you prefer to rely solely on
                   the internal counter, pass ``step=self._step``.
        """
        with self._lock:
            self._step += 1

            # During warm-up keep the shadow synchronized with the online
            # model.  This is the reset-to-online behavior used by Diffuser's
            # trainer and prevents a short horizon stage from freezing the
            # random initialization as its teacher.
            if step < self.update_after_step:
                self.copy_params_from_model_to_ema(model)
                return

            # Throttle: only update every ``update_every`` internal steps.
            if self._step % self.update_every != 0:
                return

            # On the very first qualifying update, hard-copy rather than blend,
            # so the shadow immediately reflects the current online weights
            # (avoids blending against the initial random copy).
            if not self._initialized:
                self._initialized = True
                self.copy_params_from_model_to_ema(model)
                return

            self._ema_step(model)

    def _ema_step(self, model: nn.Module) -> None:
        """Apply θ̄ ← ζ θ̄ + (1-ζ) θ for all parameters and buffers.

        Buffers (e.g. BatchNorm running statistics) are copied exactly rather
        than blended, matching standard practice.

        This method must be called while holding ``self._lock``.

        Args:
            model: The online model providing the new parameter values θ.
        """
        one_minus_decay = 1.0 - self.decay

        with torch.no_grad():
            online_params = dict(model.named_parameters())
            shadow_params = dict(self._shadow.named_parameters())

            for name, shadow_param in shadow_params.items():
                if name not in online_params:
                    # Parameter exists in shadow but not in online model;
                    # leave shadow untouched.
                    continue
                online_param = online_params[name]
                if online_param.device != shadow_param.device:
                    online_param = online_param.to(shadow_param.device)
                # θ̄ ← ζ θ̄ + (1-ζ) θ
                shadow_param.data.mul_(self.decay).add_(
                    online_param.data, alpha=one_minus_decay
                )

            # Hard-copy buffers (e.g. BatchNorm running_mean / running_var,
            # num_batches_tracked, etc.) — no blending.
            online_buffers = dict(model.named_buffers())
            shadow_buffers = dict(self._shadow.named_buffers())
            for name, shadow_buf in shadow_buffers.items():
                if name in online_buffers:
                    online_buf = online_buffers[name]
                    if online_buf.device != shadow_buf.device:
                        online_buf = online_buf.to(shadow_buf.device)
                    shadow_buf.data.copy_(online_buf.data)

    # ------------------------------------------------------------------
    # Hard initialisation
    # ------------------------------------------------------------------

    def copy_params_from_model_to_ema(self, model: nn.Module) -> None:
        """Hard-copy all parameters and buffers from the online model to the shadow.

        Overwrites the shadow weights without any blending.  Useful for:
          - Initialising the shadow at the start of training.
          - Re-synchronising after loading a checkpoint.
          - Manually resetting the shadow during curriculum changes.

        Args:
            model: The online model whose weights to copy.
        """
        with self._lock:
            with torch.no_grad():
                online_params = dict(model.named_parameters())
                shadow_params = dict(self._shadow.named_parameters())
                for name, shadow_param in shadow_params.items():
                    if name in online_params:
                        online_param = online_params[name]
                        if online_param.device != shadow_param.device:
                            online_param = online_param.to(shadow_param.device)
                        shadow_param.data.copy_(online_param.data)

                online_buffers = dict(model.named_buffers())
                shadow_buffers = dict(self._shadow.named_buffers())
                for name, shadow_buf in shadow_buffers.items():
                    if name in online_buffers:
                        online_buf = online_buffers[name]
                        if online_buf.device != shadow_buf.device:
                            online_buf = online_buf.to(shadow_buf.device)
                        shadow_buf.data.copy_(online_buf.data)

    # ------------------------------------------------------------------
    # Accessor
    # ------------------------------------------------------------------

    def get_model(self) -> nn.Module:
        """Return the EMA shadow model.

        The returned module is in ``eval()`` mode and all its parameters have
        ``requires_grad=False``.  Callers should use it inside a
        ``torch.no_grad()`` context for inference.

        Returns:
            The shadow ``nn.Module`` tracking the EMA of the online model's
            parameters.
        """
        with self._lock:
            return self._shadow

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return a serialisable state dictionary for checkpointing.

        Stores the shadow model's ``state_dict``, the internal step counter,
        and the construction hyperparameters so that the EMA object can be
        reconstructed faithfully from a checkpoint.

        Returns:
            A plain Python dict suitable for ``torch.save``.

        Example::

            torch.save({"ema": ema.state_dict(), ...}, "checkpoint.pt")
        """
        with self._lock:
            return {
                "shadow_state_dict": self._shadow.state_dict(),
                "step": self._step,
                "initialized": self._initialized,
                "decay": self.decay,
                "update_every": self.update_every,
                "update_after_step": self.update_after_step,
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore EMA state from a checkpoint dictionary.

        Loads the shadow model weights and internal counter.  The
        hyperparameters stored in ``state`` are checked against the current
        instance's values and a warning is raised on mismatch (the current
        instance values take precedence so as not to silently override
        intentional changes made at resume time).

        Args:
            state: Dictionary previously returned by ``state_dict()``.

        Raises:
            KeyError:  If ``"shadow_state_dict"`` or ``"step"`` is missing.
            RuntimeError: If the shadow ``state_dict`` is incompatible with the
                          current shadow model architecture (propagated from
                          ``nn.Module.load_state_dict``).
        """
        with self._lock:
            # Restore shadow weights (strict by default — architecture must match).
            self._shadow.load_state_dict(state["shadow_state_dict"], strict=True)

            # Restore internal step counter and initialisation flag.
            self._step = int(state["step"])
            self._initialized = bool(state.get("initialized", self._step > 0))

            # Warn on hyperparameter mismatches but do not override.
            for key in ("decay", "update_every", "update_after_step"):
                saved_val = state.get(key)
                current_val = getattr(self, key)
                if saved_val is not None and saved_val != current_val:
                    import warnings

                    warnings.warn(
                        f"EMA.load_state_dict: checkpoint has {key}={saved_val!r} "
                        f"but current instance has {key}={current_val!r}. "
                        f"Using the current instance value.",
                        UserWarning,
                        stacklevel=2,
                    )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EMA("
            f"decay={self.decay}, "
            f"update_every={self.update_every}, "
            f"update_after_step={self.update_after_step}, "
            f"step={self._step}, "
            f"initialized={self._initialized})"
        )
