"""Configuration and checkpoint contract tests for CLI entry points."""

from __future__ import annotations

import torch

from bpd.training.ema import EMA
from scripts.evaluate import _build_model, _load_model_weights
from scripts.train import _build_parser, _merge_config_and_args


def test_composed_yaml_config_reaches_argparse_schema() -> None:
    args = _merge_config_and_args(
        _build_parser(), ["--config", "configs/hopper_medium.yaml"]
    )
    assert args.env == "hopper-medium-v2"
    assert args.model_dim == 256
    assert args.diffusion_steps == 1000
    assert args.replay_refresh_freq == 5000


def test_evaluator_loads_current_ema_checkpoint_schema() -> None:
    config = {
        "obs_dim": 2,
        "act_dim": 1,
        "max_horizon": 2,
        "model_dim": 16,
        "num_heads": 2,
        "num_layers": 1,
        "dropout": 0.0,
        "diffusion_steps": 2,
        "beta_schedule": "cosine",
    }
    _, model = _build_model(config, torch.device("cpu"))
    ema = EMA(model, update_every=1, update_after_step=0)
    checkpoint = {"ema": ema.state_dict(), "score_net": model.state_dict()}
    _, fresh = _build_model(config, torch.device("cpu"))
    loaded = _load_model_weights(checkpoint, fresh, torch.device("cpu"))
    for expected, actual in zip(
        ema.get_model().state_dict().values(), loaded.state_dict().values()
    ):
        torch.testing.assert_close(actual, expected)
