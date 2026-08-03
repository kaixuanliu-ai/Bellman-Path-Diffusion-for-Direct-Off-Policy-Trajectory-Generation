"""Multi-process environment rollouts.

Environment stepping is pure Python/C on the host and is single-threaded by
default, so offline-dataset collection and Monte-Carlo ground-truth rollouts
leave every core but one idle -- and on a GPU box that host time is time the
accelerator spends waiting.  These helpers shard episodes across worker
processes so the CPU side keeps up with the device.

Both helpers preserve per-episode seeding, so results are identical to the
serial version regardless of the worker count.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def default_workers(requested: int | None = None) -> int:
    """Pick a worker count: explicit value, else the allocation's CPU count."""
    if requested is not None and requested > 0:
        return int(requested)
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return max(1, int(slurm))
    return max(1, (os.cpu_count() or 1))


def _run_shard(args):
    """Worker entry point: build the env locally and roll out assigned seeds."""
    make_env, policy_fn, seeds, ep_len, collect = args
    import gymnasium as gym  # imported inside the worker

    env = make_env() if callable(make_env) else gym.make(make_env)
    out = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        rows = []
        for t in range(ep_len):
            act = policy_fn(obs, t)
            nxt, rew, term, trunc, _ = env.step(act)
            rows.append((obs.astype(np.float32), np.asarray(act, np.float32),
                         np.float32(rew), nxt.astype(np.float32), bool(term)))
            obs = nxt
            if term or trunc:
                break
        out.append(rows if collect else _summarize(rows))
    env.close()
    return out


def _summarize(rows):
    """Reduce an episode to (initial_obs, per-step rewards)."""
    if not rows:
        return None
    return rows[0][0], np.array([r[2] for r in rows], np.float32)


def collect_dataset_parallel(
    env_id_or_factory,
    policy_fn: Callable,
    n_episodes: int,
    ep_len: int = 200,
    seed0: int = 0,
    workers: int | None = None,
) -> Dict[str, np.ndarray]:
    """Collect an offline transition dataset using all available CPU cores.

    Args:
        env_id_or_factory: Gymnasium id string or a zero-arg env factory.
        policy_fn:         ``(obs, t) -> action`` behavior policy (must be
                           picklable; a module-level function or functools
                           partial).
        n_episodes:        Number of episodes to roll out.
        ep_len:            Max steps per episode.
        seed0:             First episode seed; episode i uses ``seed0 + i``.
        workers:           Process count (defaults to the CPU allocation).

    Returns:
        Dict with ``observations``, ``actions``, ``rewards``,
        ``next_observations``, ``terminals``.
    """
    n_workers = min(default_workers(workers), max(1, n_episodes))
    seeds = np.arange(seed0, seed0 + n_episodes)
    shards = np.array_split(seeds, n_workers)
    logger.info(
        "collect_dataset_parallel: %d episodes over %d workers", n_episodes, n_workers
    )
    tasks = [(env_id_or_factory, policy_fn, s, ep_len, True) for s in shards if len(s)]
    episodes: List = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for chunk in pool.map(_run_shard, tasks):
            episodes.extend(chunk)

    obs, act, rew, nobs, term = [], [], [], [], []
    for rows in episodes:
        for o, a, r, n, d in rows:
            obs.append(o); act.append(a); rew.append(r); nobs.append(n); term.append(d)
    return {
        "observations": np.array(obs, np.float32),
        "actions": np.array(act, np.float32),
        "rewards": np.array(rew, np.float32),
        "next_observations": np.array(nobs, np.float32),
        "terminals": np.array(term, bool),
    }


def mc_return_parallel(
    env_id_or_factory,
    policy_fn: Callable,
    n_episodes: int,
    horizon: int,
    gamma: float,
    seed0: int = 10000,
    workers: int | None = None,
) -> Tuple[np.ndarray, float]:
    """Monte-Carlo truncated discounted return of a policy, across CPU cores.

    Args:
        env_id_or_factory: Gymnasium id string or zero-arg env factory.
        policy_fn:         ``(obs, t) -> action`` evaluation policy.
        n_episodes:        Rollout count.
        horizon:           Truncation horizon H.
        gamma:             Discount factor.
        seed0:             First episode seed.
        workers:           Process count (defaults to the CPU allocation).

    Returns:
        ``(initial_states, mean_return)`` where ``initial_states`` has shape
        ``(n_episodes, obs_dim)`` in episode-seed order.
    """
    n_workers = min(default_workers(workers), max(1, n_episodes))
    seeds = np.arange(seed0, seed0 + n_episodes)
    shards = np.array_split(seeds, n_workers)
    logger.info(
        "mc_return_parallel: %d episodes over %d workers", n_episodes, n_workers
    )
    tasks = [(env_id_or_factory, policy_fn, s, horizon, False) for s in shards if len(s)]
    results: List = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for chunk in pool.map(_run_shard, tasks):
            results.extend(chunk)

    disc = gamma ** np.arange(horizon, dtype=np.float64)
    s0, rets = [], []
    for item in results:
        if item is None:
            continue
        obs0, rewards = item
        s0.append(obs0)
        rets.append(float((rewards * disc[: len(rewards)]).sum()))
    return np.array(s0, np.float32), float(np.mean(rets))


def summarize_cpu(workers: int | None = None) -> str:
    """One-line description of the host parallelism actually in use."""
    return (
        f"cpu workers={default_workers(workers)} "
        f"(os.cpu_count={os.cpu_count()}, "
        f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', 'unset')})"
    )
