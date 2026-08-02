"""
Off-Policy Evaluation (OPE) script for Bellman Path Diffusion (BPD).

Loads a trained BPD checkpoint, runs OPEEvaluator.estimate_return(...) over
``--n_trajectories`` Monte-Carlo samples, prints the estimated return with a
95% bootstrap confidence interval, and optionally compares against a
ground-truth Monte-Carlo return obtained by rolling out the evaluation policy
in a live gym environment.

A JSON results file is always written to ``--output_dir``.

Length distribution diagnostics (Eq. 14) are printed and included in the JSON.

Usage
-----
python scripts/evaluate.py \\
    --checkpoint results/hopper/ckpt_H8.pt \\
    --env hopper-medium-v2 \\
    --n_trajectories 1000

Optional flags
--------------
--n_mc_episodes N   Roll out N episodes in the gym env and report the
                    empirical MC return alongside the OPE estimate.
--seed S            Global random seed (NumPy + PyTorch).
--device DEVICE     Torch device string, e.g. "cuda" or "cpu".
--output_dir DIR    Directory to write results.json (default: results/).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root on sys.path (supports running the script from any cwd)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bpd.data.dataset import Normalizer, load_d4rl_dataset  # noqa: E402
from bpd.evaluation.ope import OPEEvaluator  # noqa: E402
from bpd.models.diffusion import BlockwiseDiffusion, DDPMSchedule  # noqa: E402
from bpd.models.score_net import TrajectoryScoreNet  # noqa: E402
from bpd.utils.serialization import load_checkpoint  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bpd.evaluate")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BPD Off-Policy Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- required ---
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a .pt checkpoint file or a directory containing checkpoints.",
    )
    p.add_argument(
        "--env",
        type=str,
        required=True,
        help="D4RL environment name, e.g. 'hopper-medium-v2'.",
    )

    # --- evaluation ---
    p.add_argument(
        "--n_trajectories",
        type=int,
        default=1000,
        help="Number M of Monte-Carlo BPD trajectories for OPE.",
    )
    p.add_argument(
        "--n_mc_episodes",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set, roll out N episodes in the live gym environment and "
            "report the empirical MC return as a ground-truth baseline."
        ),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Number of trajectories generated per diffusion forward pass. "
            "Defaults to --n_trajectories (all at once). Reduce if GPU OOM."
        ),
    )

    # --- infra ---
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for NumPy and PyTorch.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory where results.json will be written.",
    )
    p.add_argument(
        "--n_bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for the 95%% CI.",
    )
    return p


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Global seed set to %d", seed)


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------


def _load_ckpt_and_config(
    checkpoint: str,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load checkpoint dict and embedded config from *checkpoint*.

    The checkpoint is expected to be a dict containing at least:
      - ``"model"``  – model state_dict (or EMA shadow).
      - ``"config"`` – hyperparameter dict (as saved by the trainer).

    Falls back gracefully when config is absent (raises a descriptive error).

    Returns:
        (ckpt_dict, config)
    """
    logger.info("Loading checkpoint: %s", checkpoint)
    ckpt: Dict[str, Any] = load_checkpoint(checkpoint, device=device)

    if "config" not in ckpt:
        # Try a sibling config.json in the same directory.
        ckpt_path = Path(checkpoint)
        if ckpt_path.is_file():
            config_candidate = ckpt_path.parent / "config.json"
        else:
            config_candidate = ckpt_path / "config.json"

        if config_candidate.exists():
            with config_candidate.open("r") as fh:
                config = json.load(fh)
            logger.info("Loaded config from %s", config_candidate)
        else:
            raise KeyError(
                "Checkpoint does not contain a 'config' key and no sibling "
                f"config.json was found.  Searched: {config_candidate}"
            )
    else:
        config = ckpt["config"]

    return ckpt, config


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _build_model(
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[BlockwiseDiffusion, TrajectoryScoreNet]:
    """Instantiate diffusion and score-network from *config*.

    Reads the following config keys (with sensible defaults):

    Top-level:
        obs_dim, act_dim, gamma, max_horizon (= H)

    config["model"]:
        model_dim, num_heads, num_layers, dropout

    config["diffusion"]:
        num_steps, beta_schedule ("linear" | "cosine")
    """
    obs_dim: int = int(config["obs_dim"])
    act_dim: int = int(config["act_dim"])
    token_dim: int = 2 + obs_dim + act_dim  # reward + obs + act + injective-phi flag
    max_horizon: int = int(config.get("max_horizon", 8))

    model_cfg: Dict[str, Any] = config.get("model", {})
    diff_cfg: Dict[str, Any] = config.get("diffusion", {})

    # --- Noise schedule ---
    T: int = int(config.get("diffusion_steps", diff_cfg.get("num_steps", 1000)))
    beta_schedule: str = config.get(
        "beta_schedule", diff_cfg.get("beta_schedule", "cosine")
    )

    if beta_schedule == "cosine":
        schedule = DDPMSchedule.make_cosine(T=T)
    elif beta_schedule == "linear":
        schedule = DDPMSchedule.make_linear(T=T)
    else:
        raise ValueError(
            f"Unknown beta_schedule '{beta_schedule}'. Choose 'cosine' or 'linear'."
        )

    # --- Diffusion wrapper ---
    # The parameterization MUST match training: an epsilon checkpoint loaded as
    # a v model (or vice versa) produces wrong samples.  Default to "v" (the
    # training default) only when the checkpoint predates this field.
    prediction_type: str = config.get(
        "prediction_type", diff_cfg.get("prediction_type", "v")
    )
    diffusion = BlockwiseDiffusion(
        schedule=schedule, token_dim=token_dim, prediction_type=prediction_type
    ).to(device)
    logger.info("Diffusion prediction_type=%s", prediction_type)

    # --- Score network ---
    score_net = TrajectoryScoreNet(
        obs_dim=obs_dim,
        act_dim=act_dim,
        token_dim=token_dim,
        model_dim=int(config.get("model_dim", model_cfg.get("model_dim", 256))),
        num_heads=int(config.get("num_heads", model_cfg.get("num_heads", 8))),
        num_layers=int(config.get("num_layers", model_cfg.get("num_layers", 6))),
        max_horizon=max_horizon,
        dropout=float(config.get("dropout", model_cfg.get("dropout", 0.0))),
    ).to(device)

    n_params = score_net.num_parameters()
    logger.info(
        "Built TrajectoryScoreNet: obs_dim=%d  act_dim=%d  token_dim=%d  "
        "model_dim=%d  max_horizon=%d  params=%s",
        obs_dim,
        act_dim,
        token_dim,
        config.get("model_dim", model_cfg.get("model_dim", 256)),
        max_horizon,
        f"{n_params:,}",
    )

    return diffusion, score_net


def _load_model_weights(
    ckpt: Dict[str, Any],
    score_net: TrajectoryScoreNet,
    device: torch.device,
) -> TrajectoryScoreNet:
    """Load model weights from *ckpt* into *score_net*.

    The current checkpoint schema stores the inference weights under
    ``ema["shadow_state_dict"]`` and the online weights under ``score_net``.
    Legacy ``ema_model``/``model`` keys remain readable.

    The loaded state dict may have been saved from a ``DataParallel`` wrapper
    (keys prefixed with ``"module."``) or from an EMA helper that stores a
    plain ``nn.Module``; both cases are handled.
    """
    candidates = []
    if isinstance(ckpt.get("ema"), dict) and "shadow_state_dict" in ckpt["ema"]:
        candidates.append(("ema.shadow_state_dict", ckpt["ema"]["shadow_state_dict"]))
    for key in ("score_net", "ema_model", "model"):
        if key in ckpt:
            candidates.append((key, ckpt[key]))

    for key, raw in candidates:
        # The EMA helper may store the full EMA object or just the state_dict.
        if isinstance(raw, dict):
            state_dict = raw
        elif hasattr(raw, "state_dict"):
            state_dict = raw.state_dict()
        else:
            logger.warning(
                "Unexpected type for ckpt['%s']: %s — skipping.", key, type(raw)
            )
            continue

        # Strip DataParallel prefix if present.
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

        score_net.load_state_dict(state_dict, strict=True)

        logger.info("Loaded model weights from ckpt['%s'].", key)
        score_net = score_net.to(device).eval()
        return score_net

    raise KeyError(
        "Checkpoint does not contain BPD model weights. "
        f"Available keys: {list(ckpt.keys())}"
    )


# ---------------------------------------------------------------------------
# Normalizer construction from dataset statistics
# ---------------------------------------------------------------------------


def _build_normalizer(
    env_name: str,
    ckpt: Dict[str, Any],
) -> Normalizer:
    """Build a dataset Normalizer.

    Preference order:
      1. Normalizer statistics embedded in the checkpoint under ``"normalizer"``.
      2. Freshly fit from the D4RL offline dataset (requires d4rl + gym).
    """
    if "normalizer" in ckpt:
        norm_state = ckpt["normalizer"]
        normalizer = Normalizer.__new__(Normalizer)
        # Restore from saved state (dict of numpy arrays / Python scalars).
        for attr, value in norm_state.items():
            setattr(
                normalizer,
                attr,
                np.array(value, dtype=np.float32)
                if not isinstance(value, float)
                else float(value),
            )
        # Restore eps if present, default otherwise.
        if not hasattr(normalizer, "eps"):
            normalizer.eps = 1e-8
        logger.info("Restored normalizer from checkpoint.")
        return normalizer

    logger.info(
        "No normalizer in checkpoint. Fitting from D4RL dataset '%s' …", env_name
    )
    data = load_d4rl_dataset(env_name)
    normalizer = Normalizer(data)
    logger.info("Normalizer fitted from dataset.")
    return normalizer


# ---------------------------------------------------------------------------
# Evaluation-policy reconstruction
# ---------------------------------------------------------------------------


def _build_eval_policy(
    policy_type: str,
    dataset: Dict[str, np.ndarray],
    normalizer: Normalizer,
    obs_dim: int,
    act_dim: int,
    device: torch.device,
):
    """Return a callable (states: Tensor) -> Tensor for the evaluation policy.

    Rebuild the exact policy family selected by ``scripts/train.py``.  Missing
    or unknown policy metadata is an error; evaluation never silently changes
    the target policy.
    """
    if policy_type == "random":

        def random_policy(states: torch.Tensor) -> torch.Tensor:
            return (
                torch.rand(states.shape[0], act_dim, device=states.device) * 2.0 - 1.0
            )

        return random_policy

    if policy_type == "dataset":
        normalized_states = torch.tensor(
            normalizer.normalize_obs(dataset["observations"]),
            dtype=torch.float32,
            device=device,
        )
        normalized_actions = torch.tensor(
            normalizer.normalize_act(dataset["actions"]),
            dtype=torch.float32,
            device=device,
        )

        def dataset_policy(states: torch.Tensor) -> torch.Tensor:
            states = states.to(device)
            best_distance = torch.full((states.shape[0],), float("inf"), device=device)
            nearest = torch.zeros(states.shape[0], dtype=torch.long, device=device)
            for start in range(0, normalized_states.shape[0], 65_536):
                distance = torch.cdist(
                    states, normalized_states[start : start + 65_536]
                ).square()
                chunk_best, chunk_index = distance.min(dim=1)
                improve = chunk_best < best_distance
                best_distance[improve] = chunk_best[improve]
                nearest[improve] = start + chunk_index[improve]
            return normalized_actions[nearest]

        return dataset_policy

    raise ValueError(
        f"checkpoint has unsupported eval_policy={policy_type!r}; "
        "evaluation policy must match training"
    )


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    per_trajectory_returns: List[float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute a two-sided (1-alpha) bootstrap confidence interval for the mean.

    Args:
        per_trajectory_returns: List of M return values, one per trajectory.
        n_bootstrap:            Number of bootstrap resamples.
        alpha:                  Significance level (default 0.05 → 95% CI).
        seed:                   Seed for the bootstrap RNG.  Passing the global
                                run seed makes the reported CI reproducible.

    Returns:
        (ci_lo, ci_hi) as Python floats.
    """
    arr = np.array(per_trajectory_returns, dtype=np.float64)
    M = len(arr)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=M, replace=True)
        boot_means[i] = sample.mean()
    ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Per-trajectory returns (needed for bootstrap)
# ---------------------------------------------------------------------------


def _collect_per_trajectory_returns(
    evaluator: OPEEvaluator,
    initial_states: torch.Tensor,
    n_trajectories: int,
    device: torch.device,
    batch_size: Optional[int] = None,
) -> Tuple[List[float], List[torch.Tensor]]:
    """Generate trajectories in batches and collect per-trajectory returns.

    Args:
        evaluator:       Configured OPEEvaluator.
        initial_states:  Tensor (N_init, obs_dim) of initial states.
        n_trajectories:  Total number of Monte-Carlo samples M.
        device:          Compute device.
        batch_size:      Number of trajectories per forward pass.
                         If None, all M trajectories are generated at once.

    Returns:
        (per_traj_returns, all_trajectories)
        * per_traj_returns – list of M floats, one undiscounted return per traj.
        * all_trajectories – list of M decoded trajectory tensors.
    """
    if batch_size is None or batch_size >= n_trajectories:
        batch_size = n_trajectories

    per_traj_returns: List[float] = []
    all_trajectories: List[torch.Tensor] = []

    remaining = n_trajectories
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        trajs = evaluator.generate_trajectories(
            initial_states=initial_states,
            n_trajectories=this_batch,
            device=device,
        )
        for traj in trajs:
            if traj.shape[0] == 0:
                ret = 0.0
            else:
                rewards = traj[:, 0]
                if evaluator.normalizer is not None and hasattr(
                    evaluator.normalizer, "unnormalize_rew"
                ):
                    rewards_np = rewards.cpu().numpy()
                    rewards_np = evaluator.normalizer.unnormalize_rew(rewards_np)
                    rewards = torch.tensor(
                        rewards_np, dtype=torch.float32, device=device
                    )
                ret = float(rewards.sum().item())
            per_traj_returns.append(ret)
            all_trajectories.append(traj)
        remaining -= this_batch
        logger.debug(
            "Generated %d / %d trajectories …",
            len(per_traj_returns),
            n_trajectories,
        )

    return per_traj_returns, all_trajectories


# ---------------------------------------------------------------------------
# Ground-truth MC evaluation via gym rollout
# ---------------------------------------------------------------------------


def _mc_ground_truth(
    env_name: str,
    eval_policy_fn,
    n_episodes: int,
    obs_dim: int,
    normalizer: Normalizer,
    gamma: float,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    """Roll out *eval_policy_fn* in the gym environment for *n_episodes*.

    Returns a dict with:
        ``"mc_mean"``  – mean discounted episode return.
        ``"mc_std"``   – std of episode returns.
        ``"mc_ci_lo"`` – 95% bootstrap CI lower bound.
        ``"mc_ci_hi"`` – 95% bootstrap CI upper bound.
        ``"n_episodes"`` – int.
    """
    try:
        import d4rl  # type: ignore[import]  # noqa: F401
        import gym  # type: ignore[import]
    except ImportError as exc:
        logger.warning("Cannot run MC ground truth: gym/d4rl not available (%s).", exc)
        return {}

    env = gym.make(env_name)
    env.seed(seed)

    episode_returns: List[float] = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        t_step = 0

        while not done:
            # Normalise observation before feeding to the policy.
            obs_norm = normalizer.normalize_obs(obs.reshape(1, -1)).reshape(-1)
            obs_t = torch.tensor(
                obs_norm, dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                act_norm = eval_policy_fn(obs_t).squeeze(0).cpu().numpy()

            # Unnormalise action before stepping in the environment.
            act = normalizer.unnormalize_act(act_norm.reshape(1, -1)).reshape(-1)

            obs, reward, done, info = env.step(act)
            ep_return += (gamma**t_step) * reward
            t_step += 1

        episode_returns.append(ep_return)

        if (ep + 1) % max(1, n_episodes // 10) == 0:
            logger.info(
                "MC rollout: %d / %d episodes  (running mean return = %.4f)",
                ep + 1,
                n_episodes,
                float(np.mean(episode_returns)),
            )

    env.close()

    arr = np.array(episode_returns, dtype=np.float64)
    ci_lo, ci_hi = _bootstrap_ci(episode_returns, n_bootstrap=2000, seed=seed)
    return {
        "mc_mean": float(arr.mean()),
        "mc_std": float(arr.std()),
        "mc_ci_lo": ci_lo,
        "mc_ci_hi": ci_hi,
        "n_episodes": n_episodes,
    }


# ---------------------------------------------------------------------------
# Length distribution reporting (Eq. 14)
# ---------------------------------------------------------------------------


def _report_length_distribution(length_dist: Dict[str, Any]) -> None:
    """Print a compact summary of the length distribution diagnostic."""
    empirical = length_dist["empirical"]
    theoretical = length_dist["theoretical"]
    kl = length_dist["kl"]
    mean_emp = length_dist["mean_empirical"]
    mean_theo = length_dist["mean_theoretical"]
    H = len(empirical)

    header = f"{'Length':>8}  {'Empirical Pr':>14}  {'Theoretical Pr':>15}"
    print("\n" + "=" * len(header))
    print("Length distribution  (Eq. 14):")
    print(header)
    print("-" * len(header))
    for length in range(1, H + 1):
        print(
            f"{length:>8}  {empirical[length - 1]:>14.6f}  "
            f"{theoretical[length - 1]:>15.6f}"
        )
    print("=" * len(header))
    print(f"  Mean length: empirical={mean_emp:.3f}  theoretical={mean_theo:.3f}")
    print(f"  KL(empirical || theoretical) = {kl:.6f} nats")
    print()


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------


def _save_results(results: Dict[str, Any], output_dir: str) -> Path:
    """Write *results* as a pretty-printed JSON file to *output_dir*."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dest = out_path / "results.json"

    # Coerce numpy arrays and tensors to Python-native types.
    def _to_serialisable(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serialisable(v) for v in obj]
        return obj

    serialisable = _to_serialisable(results)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)
        fh.write("\n")

    logger.info("Results written to %s", dest.resolve())
    return dest.resolve()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    _set_seed(args.seed)
    device = torch.device(args.device)
    logger.info("Device: %s", device)

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load checkpoint & config
    # ------------------------------------------------------------------
    ckpt, config = _load_ckpt_and_config(args.checkpoint, device)

    # Derive dimension from config, falling back to D4RL dataset name heuristics.
    obs_dim: int = int(config["obs_dim"])
    act_dim: int = int(config["act_dim"])
    gamma: float = float(config.get("gamma", 0.99))
    max_horizon: int = int(config.get("max_horizon", 8))
    token_dim_val: int = 2 + obs_dim + act_dim  # reward + obs + act + injective-phi flag

    logger.info(
        "Config: env=%s  obs_dim=%d  act_dim=%d  gamma=%.4f  H=%d",
        args.env,
        obs_dim,
        act_dim,
        gamma,
        max_horizon,
    )

    # ------------------------------------------------------------------
    # 2. Build models and load weights
    # ------------------------------------------------------------------
    diffusion, score_net = _build_model(config, device)
    score_net = _load_model_weights(ckpt, score_net, device)
    score_net.eval()

    # Wrap score_net to match the (z_t, x, t) -> noise_pred signature that
    # OPEEvaluator / BlockwiseDiffusion.p_sample_loop expects.
    # TrajectoryScoreNet.forward(z, t, x, h) — note different arg order.
    # We fix h = max_horizon at evaluation time (the diffusion is run for
    # the full horizon; padding handles shorter effective lengths).
    _h = max_horizon

    def _score_net_fn(
        z_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Adapter: (B, H, d_tok), (B, obs+act), (B,) -> (B, H, d_tok)."""
        return score_net(z_t, t, x, h=_h)

    # ------------------------------------------------------------------
    # 3. Build normalizer from dataset statistics
    # ------------------------------------------------------------------
    normalizer = _build_normalizer(args.env, ckpt)

    # ------------------------------------------------------------------
    # 4. Load dataset to obtain initial states mu_0
    # ------------------------------------------------------------------
    logger.info("Loading D4RL dataset '%s' for initial states …", args.env)
    try:
        dataset = load_d4rl_dataset(args.env)
        # mu_0 is the pool of episode-start observations.  The loader computes
        # this from the raw episode boundaries (terminals | timeouts) BEFORE the
        # qlearning conversion, since qlearning_dataset drops timeouts and would
        # otherwise collapse mu_0 to a single state.
        if "episode_start_observations" in dataset:
            initial_states_np = dataset["episode_start_observations"]
        else:  # pragma: no cover - legacy datasets without the explicit pool
            boundary = np.asarray(dataset["terminals"], dtype=bool) | np.asarray(
                dataset.get("timeouts", np.zeros_like(dataset["terminals"])),
                dtype=bool,
            )
            start_index = np.concatenate(
                (np.array([0], dtype=np.int64), np.flatnonzero(boundary) + 1)
            )
            start_index = start_index[start_index < len(dataset["observations"])]
            initial_states_np = dataset["observations"][start_index]
        # Normalise observations to match training distribution.
        initial_states_np = normalizer.normalize_obs(initial_states_np)
    except ImportError:
        raise RuntimeError(
            "D4RL is required to recover mu_0 and the configured evaluation policy"
        )

    initial_states = torch.tensor(initial_states_np, dtype=torch.float32, device=device)
    logger.info("Initial state pool size: %d", initial_states.shape[0])

    # ------------------------------------------------------------------
    # 5. Build evaluation policy
    # ------------------------------------------------------------------
    eval_policy_fn = _build_eval_policy(
        str(config.get("eval_policy", "")),
        dataset,
        normalizer,
        obs_dim,
        act_dim,
        device,
    )

    # ------------------------------------------------------------------
    # 6. Build OPEEvaluator
    # ------------------------------------------------------------------
    ope_config = {
        "horizon": max_horizon,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "gamma": gamma,
        "norm_threshold": float(config.get("norm_threshold", 0.3)),
    }

    evaluator = OPEEvaluator(
        diffusion=diffusion,
        score_net=_score_net_fn,
        schedule=diffusion.schedule,
        normalizer=normalizer,
        eval_policy_fn=eval_policy_fn,
        config=ope_config,
    )

    # ------------------------------------------------------------------
    # 7. Generate trajectories & compute per-trajectory returns
    # ------------------------------------------------------------------
    logger.info(
        "Generating %d BPD trajectories (horizon H=%d) …",
        args.n_trajectories,
        max_horizon,
    )
    t_gen_start = time.perf_counter()

    per_traj_returns, all_trajectories = _collect_per_trajectory_returns(
        evaluator=evaluator,
        initial_states=initial_states,
        n_trajectories=args.n_trajectories,
        device=device,
        batch_size=args.batch_size,
    )

    t_gen_elapsed = time.perf_counter() - t_gen_start
    logger.info(
        "Trajectory generation complete: %.2f s  (%.1f traj/s)",
        t_gen_elapsed,
        args.n_trajectories / max(t_gen_elapsed, 1e-9),
    )

    # ------------------------------------------------------------------
    # 8. OPE estimate (Eq. 59) & 95% bootstrap CI
    # ------------------------------------------------------------------
    ope_estimate = float(np.mean(per_traj_returns))
    ci_lo, ci_hi = _bootstrap_ci(
        per_traj_returns, n_bootstrap=args.n_bootstrap, alpha=0.05, seed=args.seed
    )

    print("\n" + "=" * 60)
    print("  OPE Estimate (Eq. 59):")
    print(f"    J_hat_H(pi) = {ope_estimate:.4f}")
    print(f"    95%% CI      = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    n_trajectories = {args.n_trajectories}")
    print(f"    horizon H      = {max_horizon}")
    print(f"    gamma          = {gamma}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 9. Length distribution (Eq. 14)
    # ------------------------------------------------------------------
    length_dist = evaluator.compute_length_distribution(all_trajectories)
    _report_length_distribution(length_dist)

    # ------------------------------------------------------------------
    # 10. Optional ground-truth MC return
    # ------------------------------------------------------------------
    mc_results: Dict[str, Any] = {}
    if args.n_mc_episodes is not None:
        logger.info(
            "Running %d MC episodes in '%s' for ground-truth comparison …",
            args.n_mc_episodes,
            args.env,
        )
        mc_results = _mc_ground_truth(
            env_name=args.env,
            eval_policy_fn=eval_policy_fn,
            n_episodes=args.n_mc_episodes,
            obs_dim=obs_dim,
            normalizer=normalizer,
            gamma=gamma,
            device=device,
            seed=args.seed,
        )
        if mc_results:
            mc_mean = mc_results["mc_mean"]
            mc_ci_lo = mc_results["mc_ci_lo"]
            mc_ci_hi = mc_results["mc_ci_hi"]
            ope_error = abs(ope_estimate - mc_mean)
            print("\n" + "=" * 60)
            print("  Ground-Truth MC Return:")
            print(f"    J_MC(pi)  = {mc_mean:.4f}  ± {mc_results['mc_std']:.4f}")
            print(f"    95%% CI    = [{mc_ci_lo:.4f}, {mc_ci_hi:.4f}]")
            print(f"    n_episodes = {mc_results['n_episodes']}")
            print(f"\n  |J_hat - J_MC| = {ope_error:.4f}")
            print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 11. Assemble and save results JSON
    # ------------------------------------------------------------------
    results: Dict[str, Any] = {
        "ope": {
            "estimate": ope_estimate,
            "ci_lo_95": ci_lo,
            "ci_hi_95": ci_hi,
            "n_trajectories": args.n_trajectories,
            "n_bootstrap": args.n_bootstrap,
        },
        "length_distribution": {
            "empirical": length_dist["empirical"].tolist(),
            "theoretical": length_dist["theoretical"].tolist(),
            "kl_empirical_theoretical_nats": float(length_dist["kl"]),
            "mean_empirical": float(length_dist["mean_empirical"]),
            "mean_theoretical": float(length_dist["mean_theoretical"]),
        },
        "hyperparameters": {
            "env": args.env,
            "horizon": max_horizon,
            "gamma": gamma,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "token_dim": token_dim_val,
        },
        "runtime": {
            "trajectory_generation_s": round(t_gen_elapsed, 3),
            "total_s": round(time.perf_counter() - t0, 3),
        },
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "device": str(device),
    }

    if mc_results:
        results["ground_truth_mc"] = mc_results
        if "mc_mean" in mc_results:
            results["ope"]["abs_error_vs_mc"] = abs(
                ope_estimate - mc_results["mc_mean"]
            )

    dest = _save_results(results, args.output_dir)
    logger.info("Evaluation complete.  Results saved to: %s", dest)


if __name__ == "__main__":
    main()
