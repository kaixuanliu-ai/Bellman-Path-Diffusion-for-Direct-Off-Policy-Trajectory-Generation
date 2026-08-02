"""Integration tests for the authoritative stagewise training path."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bpd.core.objectives import BellmanDiffusionLoss
from bpd.data.dataset import Normalizer, TransitionDataset
from bpd.data.replay import SuffixReplayBuffer
from bpd.evaluation.ope import OPEEvaluator
from bpd.models.diffusion import BlockwiseDiffusion, DDPMSchedule
from bpd.models.score_net import TrajectoryScoreNet
from bpd.training.ema import EMA
from bpd.training.trainer import BellmanPathDiffusionTrainer


def make_dataset() -> TransitionDataset:
    rng = np.random.default_rng(11)
    count, obs_dim, act_dim = 16, 2, 1
    data = {
        "observations": rng.normal(size=(count, obs_dim)).astype(np.float32),
        "actions": rng.normal(size=(count, act_dim)).astype(np.float32),
        "rewards": rng.normal(size=count).astype(np.float32),
        "next_observations": rng.normal(size=(count, obs_dim)).astype(np.float32),
        "terminals": np.zeros(count, dtype=bool),
    }
    return TransitionDataset(data, Normalizer(data))


def test_two_horizon_training_smoke() -> None:
    torch.manual_seed(3)
    dataset = make_dataset()
    # d_tok = reward + obs + act + injective-phi flag.
    token_dim = 2 + dataset.obs_dim + dataset.act_dim
    schedule = DDPMSchedule.make_cosine(T=2)
    diffusion = BlockwiseDiffusion(schedule, token_dim)
    model = TrajectoryScoreNet(
        dataset.obs_dim,
        dataset.act_dim,
        token_dim,
        model_dim=16,
        num_heads=2,
        num_layers=1,
        max_horizon=2,
    )
    objective = BellmanDiffusionLoss(0.6, schedule, token_dim, diffusion=diffusion)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.9, update_every=1, update_after_step=0)
    trainer = BellmanPathDiffusionTrainer(
        model,
        diffusion,
        objective,
        optimizer,
        ema,
        {
            "H": 2,
            "gamma": 0.6,
            "steps_per_horizon": 1,
            "batch_size": 4,
            "log_freq": 1,
            "save_freq": 0,
            "replay_buffer_size": 8,
            "replay_refresh_size": 2,
            "replay_refresh_freq": 0,
        },
    )

    def deterministic_policy(states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(states.shape[0], dataset.act_dim, device=states.device)

    trainer.train_all_stages(dataset, deterministic_policy, torch.device("cpu"))
    assert trainer._global_step == 2
    assert set(trainer._teachers) == {1, 2}
    assert all(
        not parameter.requires_grad for parameter in trainer._teachers[1].parameters()
    )


def test_adaln_zero_network_starts_at_zero() -> None:
    # token_dim = 2 + obs(2) + act(1) = 5 (reward + obs + act + flag).
    model = TrajectoryScoreNet(2, 1, 5, model_dim=16, num_heads=2, num_layers=2)
    output = model(
        torch.randn(3, 2, 5),
        torch.tensor([1, 2, 3]),
        torch.randn(3, 3),
        h=2,
    )
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_replay_default_key_is_exact() -> None:
    replay = SuffixReplayBuffer(4, suffix_horizon=1, token_dim=2)
    key = torch.tensor([1.0], dtype=torch.float32)
    neighboring_key = torch.nextafter(key, torch.tensor([2.0]))
    replay.add(key, torch.ones(1, 2))
    assert replay.sample_or_none(key) is not None
    assert replay.sample_or_none(neighboring_key) is None


def test_dataset_adds_absorbing_zero_reward_self_transition() -> None:
    raw = {
        "observations": np.array([[1.0, 2.0]], dtype=np.float32),
        "actions": np.array([[0.5]], dtype=np.float32),
        "rewards": np.array([3.0], dtype=np.float32),
        "next_observations": np.array([[9.0, 9.0]], dtype=np.float32),
        "terminals": np.array([True]),
    }
    normalizer = Normalizer(raw)
    dataset = TransitionDataset(raw, normalizer)
    assert len(dataset) == 2
    torch.testing.assert_close(dataset.next_states[0], torch.zeros(2))
    absorbing = dataset[1]
    torch.testing.assert_close(absorbing.state, torch.zeros(2))
    torch.testing.assert_close(absorbing.action, torch.zeros(1))
    torch.testing.assert_close(absorbing.next_state, torch.zeros(2))
    raw_zero_reward = normalizer.unnormalize_rew(
        np.array([absorbing.reward.item()], dtype=np.float32)
    )[0]
    assert raw_zero_reward == pytest.approx(0.0)


def test_decoder_keeps_zero_reward_transition_and_stops_at_flag_padding() -> None:
    # token_dim = 2 + obs(2) + act(1) = 5; layout [r, s'(2), a'(1), flag].
    schedule = DDPMSchedule.make_cosine(2)
    diffusion = BlockwiseDiffusion(schedule, token_dim=5)
    evaluator = OPEEvaluator(
        diffusion,
        lambda z, x, t: torch.zeros_like(z),
        schedule,
        normalizer=None,
        eval_policy_fn=lambda state: torch.zeros(state.shape[0], 1),
        config={"horizon": 3, "obs_dim": 2, "act_dim": 1, "gamma": 0.9},
    )
    tokens = torch.tensor(
        [
            # Zero-reward, near-zero transition: real thanks to REAL_FLAG (+1),
            # which the old ‖token‖<threshold heuristic could have misread.
            [0.0, 1.0, -1.0, 0.2, 1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0],  # padding (PAD_FLAG = -1)
            [2.0, 1.0, 1.0, 1.0, 1.0],  # must remain padding after first pad
        ]
    )
    _, real = evaluator.decode_trajectory(tokens, h=3)
    assert real.tolist() == [True, False, False]
