"""
Offline transition dataset for Bellman Path Diffusion (BPD).

Implements the offline dataset  D = {(s_i, a_i, r_i, s_i')}_{i=1}^N  from
Eq. 8 of the paper.  Each element of the dataset is a single one-step
transition – the minimal unit consumed by the Bellman path diffusion model.

Token layout (Eq. 7):
    y = (r, s', a')  ∈  Y = R × S × A

Usage
-----
>>> dataset = D4RLTransitionDataset("hopper-medium-v2", normalizer)
>>> batch = dataset[0]          # DataBatch namedtuple of normalised tensors
>>> token = dataset.sample_eval_token(0, eval_policy_fn)  # y for path building

References
----------
* Janner et al., "Planning with Diffusion" (Diffuser), NeurIPS 2022.
  – sequence dataset: diffuser/datasets/sequence.py
* Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning"
  (IQL), ICLR 2022.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.utils.data

from bpd.core.path import encode_token
from bpd.utils.arrays import DataBatch, to_torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# D4RL loader
# ---------------------------------------------------------------------------


def load_d4rl_dataset(env_name: str) -> Dict[str, np.ndarray]:
    """Load an offline dataset from D4RL.

    Returns a dictionary with the following keys and dtypes:

    ===================  ========================  ===========================
    Key                  Shape                     Description
    ===================  ========================  ===========================
    ``observations``     ``(N, obs_dim)``          Current observations s_i
    ``actions``          ``(N, act_dim)``          Actions a_i
    ``rewards``          ``(N,)``                  Rewards r_i
    ``next_observations``  ``(N, obs_dim)``        Next observations s_i'
    ``terminals``        ``(N,)`` – bool           Episode termination flags
    ``timeouts``         ``(N,)`` – bool           Timeout flags (if present)
    ===================  ========================  ===========================

    The function guards the D4RL import so that the rest of the BPD codebase
    remains usable in environments where D4RL is not installed.

    Args:
        env_name: D4RL environment name, e.g. ``"hopper-medium-v2"``.

    Returns:
        Dictionary of NumPy arrays describing N one-step transitions.

    Raises:
        ImportError: If ``d4rl`` or ``gym`` are not installed.
        RuntimeError: If the dataset returned by D4RL is missing required keys.
    """
    try:
        import d4rl  # type: ignore[import]  # noqa: F401 – registers envs
        import gym  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "D4RL and/or gym are not installed.  Install them with:\n"
            "  pip install gym==0.21.0 d4rl\n"
            f"Original error: {exc}"
        ) from exc

    env = gym.make(env_name)
    raw: Dict[str, np.ndarray] = env.get_dataset()

    # Validate required keys
    required = {"observations", "actions", "rewards", "next_observations", "terminals"}
    missing = required - set(raw.keys())
    if missing:
        raise RuntimeError(f"D4RL dataset for '{env_name}' is missing keys: {missing}")

    N = raw["observations"].shape[0]
    logger.info(
        "Loaded D4RL dataset '%s': %d transitions, obs_dim=%d, act_dim=%d",
        env_name,
        N,
        raw["observations"].shape[1],
        raw["actions"].shape[1],
    )

    dataset: Dict[str, np.ndarray] = {
        "observations": raw["observations"].astype(np.float32),
        "actions": raw["actions"].astype(np.float32),
        "rewards": raw["rewards"].astype(np.float32).reshape(N),
        "next_observations": raw["next_observations"].astype(np.float32),
        "terminals": raw["terminals"].astype(bool).reshape(N),
    }

    # timeouts are present in most D4RL datasets but not all
    if "timeouts" in raw:
        dataset["timeouts"] = raw["timeouts"].astype(bool).reshape(N)
    else:
        dataset["timeouts"] = np.zeros(N, dtype=bool)

    return dataset


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------


class Normalizer:
    """Per-feature mean/std normaliser fitted on an offline dataset.

    Fits statistics on ``observations``, ``actions``, and ``rewards`` and
    exposes a ``normalize`` method that maps each field to zero mean, unit
    variance.  Normalization is performed in float32.

    Args:
        data: Dictionary containing ``"observations"``, ``"actions"``,
              and ``"rewards"`` as NumPy arrays.
        eps:  Small constant added to the standard deviation for numerical
              stability.
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        eps: float = 1e-8,
    ) -> None:
        self.eps = eps

        obs = data["observations"].astype(np.float32)
        act = data["actions"].astype(np.float32)
        rew = data["rewards"].astype(np.float32).reshape(-1)

        self.obs_mean: np.ndarray = obs.mean(axis=0)
        self.obs_std: np.ndarray = obs.std(axis=0)

        self.act_mean: np.ndarray = act.mean(axis=0)
        self.act_std: np.ndarray = act.std(axis=0)

        self.rew_mean: float = float(rew.mean())
        self.rew_std: float = float(rew.std())

    # ------------------------------------------------------------------
    # Normalise methods
    # ------------------------------------------------------------------

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalise observations to zero mean / unit std."""
        return (obs - self.obs_mean) / (self.obs_std + self.eps)

    def normalize_act(self, act: np.ndarray) -> np.ndarray:
        """Normalise actions to zero mean / unit std."""
        return (act - self.act_mean) / (self.act_std + self.eps)

    def normalize_rew(self, rew: np.ndarray) -> np.ndarray:
        """Normalise rewards to zero mean / unit std."""
        return (rew - self.rew_mean) / (self.rew_std + self.eps)

    # ------------------------------------------------------------------
    # Unnormalise methods (convenience for evaluation)
    # ------------------------------------------------------------------

    def unnormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Invert observation normalisation."""
        return obs * (self.obs_std + self.eps) + self.obs_mean

    def unnormalize_act(self, act: np.ndarray) -> np.ndarray:
        """Invert action normalisation."""
        return act * (self.act_std + self.eps) + self.act_mean

    def unnormalize_rew(self, rew: np.ndarray) -> np.ndarray:
        """Invert reward normalisation."""
        return rew * (self.rew_std + self.eps) + self.rew_mean


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TransitionDataset(torch.utils.data.Dataset):
    """PyTorch Dataset of one-step offline transitions.

    Stores the dataset  D = {(s_i, a_i, r_i, s_i')}_{i=1}^N  (Eq. 8) as
    normalised float32 tensors on the requested *device*.

    Each ``__getitem__`` call returns a :class:`~bpd.utils.arrays.DataBatch`
    namedtuple of tensors.  The optional ``sample_eval_token`` method computes
    the full BPD token  y = (r, s', a')  (Eq. 7) by querying the evaluation
    policy  π_e  at the next state  s'.

    Args:
        data:            Dictionary with keys

                         * ``"observations"``      – ``(N, obs_dim)`` float32
                         * ``"actions"``           – ``(N, act_dim)`` float32
                         * ``"rewards"``           – ``(N,)``         float32
                         * ``"next_observations"`` – ``(N, obs_dim)`` float32
                         * ``"terminals"``         – ``(N,)``         bool

        normalizer:      Fitted :class:`Normalizer` instance used to
                         standardise observations, actions, and rewards.
        eval_policy_fn:  Optional callable  ``(state: Tensor) -> Tensor``
                         implementing the evaluation policy  π_e.  When
                         provided, :meth:`sample_eval_token` uses it to draw
                         a' ~ π_e(·|s').
        device:          Target device for all stored tensors (default ``"cpu"``).
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        normalizer: Normalizer,
        eval_policy_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Union[str, torch.device] = "cpu",
        add_absorbing_transition: bool = True,
    ) -> None:
        super().__init__()

        self.normalizer = normalizer
        self.eval_policy_fn = eval_policy_fn
        self.device = torch.device(device)

        # ------------------------------------------------------------------ #
        # Normalise and cast to float32 tensors                               #
        # ------------------------------------------------------------------ #
        obs = normalizer.normalize_obs(data["observations"].astype(np.float32))
        act = normalizer.normalize_act(data["actions"].astype(np.float32))
        rew = normalizer.normalize_rew(data["rewards"].astype(np.float32).reshape(-1))
        next_obs = normalizer.normalize_obs(
            data["next_observations"].astype(np.float32)
        )
        terminals = data["terminals"].astype(np.float32).reshape(-1)
        self.num_logged_transitions: int = int(obs.shape[0])

        # Paper Section 3 represents environment termination by one canonical
        # absorbing state-action pair.  Zero is reserved for that pair in the
        # normalized conditioning space.  Terminal successors are redirected
        # to it, and a zero-reward absorbing self-transition is added so the
        # conditional kernel is covered during training.
        terminal_mask = terminals.astype(bool)
        next_obs[terminal_mask] = 0.0
        if add_absorbing_transition:
            zero_reward = normalizer.normalize_rew(
                np.zeros(1, dtype=np.float32)
            ).astype(np.float32)
            obs = np.concatenate((obs, np.zeros((1, obs.shape[1]), dtype=np.float32)))
            act = np.concatenate((act, np.zeros((1, act.shape[1]), dtype=np.float32)))
            rew = np.concatenate((rew, zero_reward.reshape(1)))
            next_obs = np.concatenate(
                (next_obs, np.zeros((1, next_obs.shape[1]), dtype=np.float32))
            )
            terminals = np.concatenate((terminals, np.ones(1, dtype=np.float32)))

        # Store on the target device as contiguous float32 tensors.
        self._states: torch.Tensor = to_torch(
            obs, dtype=torch.float32, device=self.device
        )
        self._actions: torch.Tensor = to_torch(
            act, dtype=torch.float32, device=self.device
        )
        self._rewards: torch.Tensor = to_torch(
            rew, dtype=torch.float32, device=self.device
        )
        self._next_states: torch.Tensor = to_torch(
            next_obs, dtype=torch.float32, device=self.device
        )
        self._dones: torch.Tensor = to_torch(
            terminals, dtype=torch.float32, device=self.device
        )

        self._N: int = self._states.shape[0]
        self.obs_dim: int = self._states.shape[1]
        self.act_dim: int = self._actions.shape[1]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of transitions N in the dataset."""
        return self._N

    def __getitem__(self, idx: int) -> DataBatch:
        """Return the normalised transition at index *idx*.

        Args:
            idx: Integer index in ``[0, N)``.

        Returns:
            :class:`~bpd.utils.arrays.DataBatch` namedtuple with fields:

            * ``state``      – ``(obs_dim,)`` float32 tensor
            * ``action``     – ``(act_dim,)`` float32 tensor
            * ``reward``     – scalar float32 tensor ``()``
            * ``next_state`` – ``(obs_dim,)`` float32 tensor
            * ``done``       – scalar float32 tensor ``()``
        """
        return DataBatch(
            state=self._states[idx],
            action=self._actions[idx],
            reward=self._rewards[idx],
            next_state=self._next_states[idx],
            done=self._dones[idx],
        )

    # ------------------------------------------------------------------
    # Evaluation token  (Eq. 7)
    # ------------------------------------------------------------------

    def sample_eval_token(
        self,
        idx: int,
        eval_policy_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Sample the BPD token  y = (r, s', a')  from the evaluation policy.

        Implements the token construction from Eq. 7 of the paper:

            y = (r_i, s_i', a')   where   a' ~ π_e(· | s_i')

        The evaluation policy  π_e  is taken from *eval_policy_fn* if
        provided, falling back to ``self.eval_policy_fn`` if set at
        construction time.  At least one must be non-``None``.

        Args:
            idx:            Index of the transition in ``[0, N)``.
            eval_policy_fn: Optional callable ``(state: Tensor) -> Tensor``
                            that samples an action from the evaluation policy
                            given a *normalised* next-state tensor of shape
                            ``(obs_dim,)``.  The returned action must have shape
                            ``(act_dim,)`` and be expressed in the *normalised*
                            action space so that the token is consistent with
                            the dataset's normalisation.

        Returns:
            Token tensor  y  of shape  ``(2 + obs_dim + act_dim,)``  with
            layout ``[r | s' | a' | flag]`` where the trailing coordinate is the
            injective-φ real-token flag (:data:`bpd.core.path.REAL_FLAG`).

        Raises:
            ValueError: If no evaluation policy is available.
        """
        policy = eval_policy_fn if eval_policy_fn is not None else self.eval_policy_fn
        if policy is None:
            raise ValueError(
                "sample_eval_token requires an evaluation policy function.  "
                "Pass eval_policy_fn at construction or as an argument."
            )

        reward: torch.Tensor = self._rewards[idx].reshape(1)  # (1,)
        next_state: torch.Tensor = self._next_states[idx]  # (obs_dim,)

        with torch.no_grad():
            if bool(self._dones[idx].item()):
                next_action = torch.zeros(
                    self.act_dim, dtype=torch.float32, device=self.device
                )
            else:
                next_action = policy(next_state)  # (act_dim,)

        next_action = next_action.to(dtype=torch.float32, device=self.device).reshape(
            -1
        )

        # y = φ([r, s', a']) with trailing REAL_FLAG – Eq. 7 (injective φ)
        return encode_token(reward, next_state, next_action)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def states(self) -> torch.Tensor:
        """All normalised state tensors ``(N, obs_dim)``."""
        return self._states

    @property
    def actions(self) -> torch.Tensor:
        """All normalised action tensors ``(N, act_dim)``."""
        return self._actions

    @property
    def rewards(self) -> torch.Tensor:
        """All normalised reward tensors ``(N,)``."""
        return self._rewards

    @property
    def next_states(self) -> torch.Tensor:
        """All normalised next-state tensors ``(N, obs_dim)``."""
        return self._next_states

    @property
    def dones(self) -> torch.Tensor:
        """Terminal flags as float32 ``(N,)``."""
        return self._dones


# ---------------------------------------------------------------------------
# D4RL convenience subclass
# ---------------------------------------------------------------------------


class D4RLTransitionDataset(TransitionDataset):
    """Thin wrapper around :class:`TransitionDataset` that loads data from D4RL.

    Calls :func:`load_d4rl_dataset` to fetch the offline dataset and then
    constructs a :class:`Normalizer` from that data before passing both to
    the parent constructor.

    Args:
        env_name:        D4RL environment name, e.g. ``"hopper-medium-v2"``.
        normalizer:      Pre-fitted :class:`Normalizer`.  If ``None`` a new
                         normalizer is fitted from the loaded dataset.
        eval_policy_fn:  Optional evaluation policy callable.
        device:          Target device for stored tensors (default ``"cpu"``).

    Example::

        from bpd.data.dataset import D4RLTransitionDataset, Normalizer

        dataset = D4RLTransitionDataset("hopper-medium-v2")
        batch = dataset[0]   # DataBatch of normalised torch tensors
    """

    def __init__(
        self,
        env_name: str,
        normalizer: Optional[Normalizer] = None,
        eval_policy_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        data = load_d4rl_dataset(env_name)

        if normalizer is None:
            normalizer = Normalizer(data)

        self.env_name = env_name

        super().__init__(
            data=data,
            normalizer=normalizer,
            eval_policy_fn=eval_policy_fn,
            device=device,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"D4RLTransitionDataset("
            f"env_name={self.env_name!r}, "
            f"N={self._N}, "
            f"obs_dim={self.obs_dim}, "
            f"act_dim={self.act_dim})"
        )
