"""
Main training entry point for Bellman Path Diffusion (BPD).

Usage
-----
Basic (all defaults):
    python scripts/train.py --env hopper-medium-v2

With overrides:
    python scripts/train.py \
        --env hopper-medium-v2 \
        --gamma 0.99 \
        --max_horizon 8 \
        --steps_per_horizon 100000 \
        --batch_size 256 \
        --lr 3e-4 \
        --seed 42 \
        --device cuda \
        --log_dir runs/hopper \
        --eval_policy dataset

With a YAML config (individual flags override config values):
    python scripts/train.py --config configs/hopper.yaml --seed 0

eval_policy choices
-------------------
dataset  : Use the dataset action at s' as a proxy for pi_e.
           Implements behavior-cloning style evaluation — useful for
           debugging without a real policy.
random   : Sample a uniformly random action in [-1, 1]^{act_dim}.
argmax   : Placeholder for a learned greedy policy.  Falls back to
           random until a real policy is plugged in.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project root on sys.path (allows running without pip install)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Optional dependency stubs — must be installed before any BPD imports
# because trainer.py performs a top-level `from torch.utils.tensorboard
# import SummaryWriter`.  We provide a no-op stub when tensorboard is absent.
# ---------------------------------------------------------------------------
try:
    import tensorboard  # noqa: F401  -- just test availability
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False
    # Inject a minimal stub so the trainer module can be imported without error.
    import types as _types

    _tb_stub = _types.ModuleType("tensorboard")
    sys.modules.setdefault("tensorboard", _tb_stub)

    _tb_summary_stub = _types.ModuleType("tensorboard.summary")
    sys.modules.setdefault("tensorboard.summary", _tb_summary_stub)

    _tw_stub = _types.ModuleType("torch.utils.tensorboard")

    class _NoOpSummaryWriter:  # noqa: D101
        """No-op stand-in for SummaryWriter when tensorboard is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def add_scalar(self, *args: Any, **kwargs: Any) -> None:
            pass

        def close(self) -> None:
            pass

    _tw_stub.SummaryWriter = _NoOpSummaryWriter  # type: ignore[attr-defined]
    sys.modules.setdefault("torch.utils.tensorboard", _tw_stub)

from bpd.core.path import token_dim as compute_token_dim
from bpd.data.dataset import D4RLTransitionDataset, TransitionDataset
from bpd.data.normalizer import GaussianNormalizer
from bpd.models.diffusion import BlockwiseDiffusion, DDPMSchedule
from bpd.models.score_net import TrajectoryScoreNet
from bpd.training.ema import EMA
from bpd.training.trainer import BellmanPathDiffusionTrainer
from bpd.core.objectives import BellmanDiffusionLoss
from bpd.utils.serialization import save_checkpoint, save_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(log_dir: str, level: int = logging.INFO) -> None:
    """Configure root logger with console + rotating file handlers."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(str(log_path / "train.log"), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------


def set_random_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN mode (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a Bellman Path Diffusion model on an offline RL dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Config file (optional)
    # ------------------------------------------------------------------
    p.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a YAML file containing default values for any of the "
            "arguments below.  Command-line flags take precedence."
        ),
    )

    # ------------------------------------------------------------------
    # Dataset / environment
    # ------------------------------------------------------------------
    p.add_argument(
        "--env",
        type=str,
        default="hopper-medium-v2",
        help="D4RL environment / dataset name.",
    )

    # ------------------------------------------------------------------
    # BPD hyper-parameters
    # ------------------------------------------------------------------
    p.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Geometric discount / Bernoulli continuation probability gamma in (0, 1).",
    )
    p.add_argument(
        "--max_horizon",
        type=int,
        default=8,
        help="Maximum path horizon H (number of BPD training stages).",
    )
    p.add_argument(
        "--steps_per_horizon",
        type=int,
        default=100_000,
        help="Number of gradient steps per horizon stage.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size (number of transitions per gradient step).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Adam learning rate.",
    )

    # ------------------------------------------------------------------
    # Diffusion / score network
    # ------------------------------------------------------------------
    p.add_argument(
        "--diffusion_steps",
        type=int,
        default=1000,
        help="Number of discrete DDPM diffusion steps T.",
    )
    p.add_argument(
        "--beta_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
        help="Noise schedule for the forward diffusion process.",
    )
    p.add_argument(
        "--model_dim",
        type=int,
        default=256,
        help="Transformer hidden dimension for the score network.",
    )
    p.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads in the score network.",
    )
    p.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of AdaLN transformer blocks.",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability inside the score network.",
    )

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------
    p.add_argument(
        "--ema_decay",
        type=float,
        default=0.995,
        help="EMA decay rate for the shadow teacher network.",
    )
    p.add_argument(
        "--ema_update_every",
        type=int,
        default=10,
        help="Apply EMA update once every this many gradient steps.",
    )
    p.add_argument(
        "--ema_update_after_step",
        type=int,
        default=100,
        help="Begin EMA updates only after this many gradient steps.",
    )

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------
    p.add_argument(
        "--replay_buffer_size",
        type=int,
        default=50_000,
        help="Maximum number of (x', W_+) entries in the suffix replay buffer.",
    )

    # ------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------
    p.add_argument(
        "--log_dir",
        type=str,
        default="runs/bpd",
        help="Directory for logs, TensorBoard events, and checkpoints.",
    )
    p.add_argument(
        "--log_freq",
        type=int,
        default=1_000,
        help="Log training metrics every this many gradient steps.",
    )
    p.add_argument(
        "--save_freq",
        type=int,
        default=10_000,
        help="Save a checkpoint every this many gradient steps.",
    )

    # ------------------------------------------------------------------
    # Evaluation policy
    # ------------------------------------------------------------------
    p.add_argument(
        "--eval_policy",
        type=str,
        default="dataset",
        choices=["dataset", "random", "argmax"],
        help=(
            "Strategy used to sample next-actions a' ~ pi_e(.|s') during training.\n"
            "  dataset : Use the stored dataset action at the next state (BC proxy).\n"
            "  random  : Sample uniformly in [-1, 1]^act_dim.\n"
            "  argmax  : Placeholder for a learned greedy policy (falls back to random)."
        ),
    )

    # ------------------------------------------------------------------
    # Device / reproducibility
    # ------------------------------------------------------------------
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed for reproducibility.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "PyTorch device string, e.g. 'cuda', 'cuda:1', 'cpu'. "
            "Auto-selects CUDA if available when not specified."
        ),
    )

    return p


def _load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML file and return its top-level mapping as a dict."""
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to use --config.  "
            "Install it with:  pip install pyyaml"
        ) from exc

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(
            f"YAML config at '{path}' must be a mapping at the top level; "
            f"got {type(cfg).__name__}"
        )
    return cfg


def _merge_config_and_args(
    parser: argparse.ArgumentParser,
    argv: Optional[list] = None,
) -> argparse.Namespace:
    """Parse args and merge with an optional YAML config.

    Priority (highest wins): CLI flags > YAML config > argparse defaults.
    """
    args = parser.parse_args(argv)

    if args.config is not None:
        yaml_cfg = _load_yaml_config(args.config)
        # For each key in the YAML config, set it on `args` only if the user
        # did NOT explicitly pass the corresponding CLI flag.
        defaults = {a.dest: a.default for a in parser._actions}
        for key, value in yaml_cfg.items():
            if not hasattr(args, key):
                logger.warning("YAML config key '%s' is not a recognised argument; skipping.", key)
                continue
            # If the current value equals the parser default, the user did not
            # pass the flag explicitly, so the YAML value takes precedence.
            if getattr(args, key) == defaults.get(key):
                setattr(args, key, value)

    return args


# ---------------------------------------------------------------------------
# Eval policy factories
# ---------------------------------------------------------------------------


def _make_eval_policy(
    policy_type: str,
    dataset: TransitionDataset,
    device: torch.device,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build an evaluation policy callable  (next_state: Tensor) -> action: Tensor.

    The returned callable accepts either a single state (obs_dim,) or a batch
    (B, obs_dim), and returns the corresponding action(s).

    Args:
        policy_type: One of 'dataset', 'random', or 'argmax'.
        dataset:     The offline dataset (used by 'dataset' policy).
        device:      Compute device for returned tensors.

    Returns:
        Callable mapping state tensor(s) to action tensor(s).
    """
    act_dim: int = dataset.act_dim

    if policy_type == "dataset":
        # Behavior-cloning proxy: look up the nearest state in the dataset by
        # L2 distance and return its stored action.  This is a cheap stand-in
        # for a real policy that lets us test the full training loop without
        # needing a policy network.
        #
        # For efficiency we precompute the state matrix on the target device.
        all_states: torch.Tensor = dataset.states.to(device)   # (N, obs_dim)
        all_actions: torch.Tensor = dataset.actions.to(device) # (N, act_dim)

        def dataset_policy(state: torch.Tensor) -> torch.Tensor:
            """Return the dataset action corresponding to the nearest stored state."""
            state = state.to(device)
            batched = state.dim() == 2  # (B, obs_dim) vs (obs_dim,)
            if not batched:
                state = state.unsqueeze(0)  # (1, obs_dim)

            # L2 nearest-neighbour (approximate behavior cloning)
            # dists: (B, N)
            diffs = all_states.unsqueeze(0) - state.unsqueeze(1)  # (B, N, obs_dim)
            dists = (diffs ** 2).sum(dim=-1)  # (B, N)
            nearest_idx = dists.argmin(dim=-1)  # (B,)
            actions = all_actions[nearest_idx]  # (B, act_dim)

            return actions if batched else actions.squeeze(0)

        return dataset_policy

    elif policy_type == "random":
        def random_policy(state: torch.Tensor) -> torch.Tensor:
            """Sample a uniformly random action in [-1, 1]^act_dim."""
            state = state.to(device)
            if state.dim() == 1:
                return torch.rand(act_dim, device=device) * 2.0 - 1.0
            B = state.shape[0]
            return torch.rand(B, act_dim, device=device) * 2.0 - 1.0

        return random_policy

    elif policy_type == "argmax":
        # Placeholder for a trained greedy policy.
        # TODO: Replace with a real learned policy (e.g., IQL actor).
        logger.warning(
            "--eval_policy=argmax: No learned policy loaded; "
            "falling back to uniform random actions."
        )

        def argmax_policy(state: torch.Tensor) -> torch.Tensor:
            """Placeholder greedy policy (currently uniform random)."""
            state = state.to(device)
            if state.dim() == 1:
                return torch.rand(act_dim, device=device) * 2.0 - 1.0
            B = state.shape[0]
            return torch.rand(B, act_dim, device=device) * 2.0 - 1.0

        return argmax_policy

    else:
        raise ValueError(
            f"Unknown eval_policy '{policy_type}'. "
            "Choose from 'dataset', 'random', or 'argmax'."
        )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Full training pipeline for Bellman Path Diffusion.

    1. Set seeds.
    2. Load offline dataset and fit normalizer.
    3. Build score network, DDPM schedule, blockwise diffusion, loss.
    4. Build EMA, optimizer, and trainer.
    5. Call trainer.train_all_stages(...).
    6. Save final checkpoint and config.

    Args:
        args: Parsed namespace from :func:`_merge_config_and_args`.
    """
    # ------------------------------------------------------------------
    # 1. Seeds and device
    # ------------------------------------------------------------------
    set_random_seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # 2. Logging directory
    # ------------------------------------------------------------------
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Load dataset and build normalizer
    # ------------------------------------------------------------------
    logger.info("Loading D4RL dataset: %s", args.env)
    dataset = D4RLTransitionDataset(
        env_name=args.env,
        normalizer=None,   # fitted automatically from the loaded data
        eval_policy_fn=None,
        device=device,
    )
    obs_dim: int = dataset.obs_dim
    act_dim: int = dataset.act_dim
    d_tok: int = compute_token_dim(obs_dim, act_dim)  # 1 + obs_dim + act_dim

    logger.info(
        "Dataset loaded: N=%d, obs_dim=%d, act_dim=%d, token_dim=%d",
        len(dataset),
        obs_dim,
        act_dim,
        d_tok,
    )

    # ------------------------------------------------------------------
    # 4. Build the evaluation policy
    # ------------------------------------------------------------------
    eval_policy_fn = _make_eval_policy(args.eval_policy, dataset, device)
    logger.info("Evaluation policy: %s", args.eval_policy)

    # ------------------------------------------------------------------
    # 5. Build score network
    # ------------------------------------------------------------------
    score_net = TrajectoryScoreNet(
        obs_dim=obs_dim,
        act_dim=act_dim,
        token_dim=d_tok,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_horizon=args.max_horizon,
        dropout=args.dropout,
    ).to(device)

    n_params = score_net.num_parameters()
    logger.info(
        "TrajectoryScoreNet: model_dim=%d, num_heads=%d, num_layers=%d, "
        "max_horizon=%d, params=%s",
        args.model_dim,
        args.num_heads,
        args.num_layers,
        args.max_horizon,
        f"{n_params:,}",
    )

    # ------------------------------------------------------------------
    # 6. Build DDPM schedule and blockwise diffusion
    # ------------------------------------------------------------------
    if args.beta_schedule == "cosine":
        schedule = DDPMSchedule.make_cosine(T=args.diffusion_steps)
    else:
        schedule = DDPMSchedule.make_linear(T=args.diffusion_steps)

    diffusion = BlockwiseDiffusion(schedule=schedule, token_dim=d_tok).to(device)

    logger.info(
        "DDPMSchedule: T=%d, beta_schedule=%s",
        args.diffusion_steps,
        args.beta_schedule,
    )

    # ------------------------------------------------------------------
    # 7. Build Bellman diffusion loss
    # ------------------------------------------------------------------
    loss_fn = BellmanDiffusionLoss(
        gamma=args.gamma,
        schedule=schedule,
        token_dim=d_tok,
        loss_weight_fn=None,  # uniform lambda(t) = 1 (Ho et al. 2020 Eq. 14)
    )

    # ------------------------------------------------------------------
    # 8. Build EMA
    # ------------------------------------------------------------------
    ema = EMA(
        model=score_net,
        decay=args.ema_decay,
        update_every=args.ema_update_every,
        update_after_step=args.ema_update_after_step,
    )

    # ------------------------------------------------------------------
    # 9. Build optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(score_net.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # 10. Build trainer config dict
    # ------------------------------------------------------------------
    trainer_config: Dict[str, Any] = {
        "gamma": args.gamma,
        "H": args.max_horizon,
        "steps_per_horizon": args.steps_per_horizon,
        "batch_size": args.batch_size,
        "log_freq": args.log_freq,
        "save_freq": args.save_freq,
        "lr": args.lr,
        "ema_decay": args.ema_decay,
        "replay_buffer_size": args.replay_buffer_size,
        "diffusion_steps": args.diffusion_steps,
        # checkpoint destination embedded in trainer config so periodic saves
        # land in the right directory.
        "_ckpt_dir": str(ckpt_dir),
    }

    # ------------------------------------------------------------------
    # 11. Build trainer
    # ------------------------------------------------------------------
    # The trainer exposes train_all_stages which orchestrates Algorithm 1.
    # We wrap loss_fn to match the trainer's LossFn signature
    # (pred, target, t) -> scalar.  The BellmanDiffusionLoss.forward has a
    # richer signature; the trainer internally calls diffusion helpers and
    # uses a simpler MSE loss_fn hook for the score matching step.
    # We pass None so the trainer uses its default MSE loss (consistent with
    # the Bellman objective used in trainer._build_training_targets).
    trainer = BellmanPathDiffusionTrainer(
        score_net=score_net,
        diffusion=diffusion,
        schedule=schedule,
        loss_fn=None,   # use trainer's default unweighted MSE
        optimizer=optimizer,
        ema=ema,
        config=trainer_config,
    )

    # Override the default checkpoint path helper to use our log_dir.
    def _ckpt_path_override(h: int, step: int) -> str:
        return str(ckpt_dir / f"bpd_h{h:02d}_step{step:07d}.pt")

    trainer._default_ckpt_path = staticmethod(_ckpt_path_override)  # type: ignore[method-assign]

    # ------------------------------------------------------------------
    # 12. Save run config before training starts
    # ------------------------------------------------------------------
    full_config: Dict[str, Any] = {
        "env": args.env,
        "gamma": args.gamma,
        "max_horizon": args.max_horizon,
        "steps_per_horizon": args.steps_per_horizon,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "diffusion_steps": args.diffusion_steps,
        "beta_schedule": args.beta_schedule,
        "model_dim": args.model_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "ema_decay": args.ema_decay,
        "ema_update_every": args.ema_update_every,
        "ema_update_after_step": args.ema_update_after_step,
        "replay_buffer_size": args.replay_buffer_size,
        "log_dir": str(log_dir),
        "log_freq": args.log_freq,
        "save_freq": args.save_freq,
        "eval_policy": args.eval_policy,
        "seed": args.seed,
        "device": str(device),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "token_dim": d_tok,
        "score_net_params": n_params,
    }
    save_config(full_config, log_dir / "config.json")
    logger.info("Config saved to %s", log_dir / "config.json")

    # ------------------------------------------------------------------
    # 13. Optional TensorBoard writer
    # ------------------------------------------------------------------
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]
        writer = SummaryWriter(log_dir=str(log_dir / "tb"))
        logger.info("TensorBoard logging enabled: %s", log_dir / "tb")
    except ImportError:
        logger.info(
            "TensorBoard not available (pip install tensorboard to enable). "
            "Continuing without TB logging."
        )

    # ------------------------------------------------------------------
    # 14. Train all H stages (Algorithm 1)
    # ------------------------------------------------------------------
    logger.info(
        "Starting BPD training: env=%s, H=%d, steps_per_horizon=%d, "
        "total_steps=%d",
        args.env,
        args.max_horizon,
        args.steps_per_horizon,
        args.max_horizon * args.steps_per_horizon,
    )

    try:
        trainer.train_all_stages(
            dataset=dataset,
            eval_policy_fn=eval_policy_fn,
            device=device,
            writer=writer,
        )
    finally:
        if writer is not None:
            writer.close()

    # ------------------------------------------------------------------
    # 15. Save final checkpoint
    # ------------------------------------------------------------------
    final_ckpt_path = ckpt_dir / "final.pt"
    final_state: Dict[str, Any] = {
        "score_net": score_net.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": trainer._global_step,
        "config": full_config,
    }
    save_checkpoint(final_state, path=ckpt_dir, filename="final.pt")
    logger.info("Final checkpoint saved: %s", final_ckpt_path)

    # Also write the config next to the final checkpoint for convenience.
    save_config(full_config, ckpt_dir / "config.json")

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> None:
    """Parse arguments and launch training."""
    parser = _build_parser()
    args = _merge_config_and_args(parser, argv)

    # Set up logging BEFORE anything else so that early log calls are captured.
    _setup_logging(args.log_dir)

    logger.info("=" * 72)
    logger.info("Bellman Path Diffusion — Training Script")
    logger.info("=" * 72)
    logger.info("Arguments:\n%s", "\n".join(f"  {k}={v}" for k, v in vars(args).items()))

    train(args)


if __name__ == "__main__":
    main()
