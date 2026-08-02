"""Equation-level tests for the Bellman DDPM objective."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from bpd.core.objectives import BellmanDiffusionLoss, get_loss_weight
from bpd.models.diffusion import DDPMSchedule
from bpd.utils.arrays import DataBatch

OBS_DIM = 3
ACT_DIM = 2
# d_tok = reward + obs + act + injective-phi flag (bpd.core.path.token_dim).
TOKEN_DIM = 2 + OBS_DIM + ACT_DIM


def make_batch(batch_size: int = 8) -> DataBatch:
    generator = torch.Generator().manual_seed(7)
    return DataBatch(
        state=torch.randn(batch_size, OBS_DIM, generator=generator),
        action=torch.randn(batch_size, ACT_DIM, generator=generator),
        reward=torch.randn(batch_size, generator=generator),
        next_state=torch.randn(batch_size, OBS_DIM, generator=generator),
        done=torch.zeros(batch_size),
    )


def make_objective(gamma: float = 0.8, steps: int = 4) -> BellmanDiffusionLoss:
    return BellmanDiffusionLoss(
        gamma=gamma,
        schedule=DDPMSchedule.make_cosine(steps),
        token_dim=TOKEN_DIM,
    )


def zero_teacher(z_t, t, x, h):
    return torch.zeros_like(z_t)


def test_stop_target_is_sampled_noise() -> None:
    objective = make_objective()
    batch = make_batch(3)
    next_action = torch.randn(3, ACT_DIM)
    noise = torch.randn(3, 4, TOKEN_DIM)
    target = objective.build_targets(
        None,
        batch,
        4,
        next_action=next_action,
        timesteps=torch.tensor([1, 2, 3]),
        continuation=torch.zeros(3, dtype=torch.bool),
        noise=noise,
    )
    torch.testing.assert_close(target.target_noise, noise)


def test_continuation_is_head_noise_plus_unscaled_teacher_noise() -> None:
    gamma = 0.25
    objective = make_objective(gamma=gamma, steps=2)
    batch = make_batch(2)
    next_action = torch.randn(2, ACT_DIM)
    noise = torch.randn(2, 3, TOKEN_DIM)

    def constant_teacher(z_t, t, x, h):
        return torch.full_like(z_t, 3.0)

    target = objective.build_targets(
        constant_teacher,
        batch,
        3,
        next_action=next_action,
        timesteps=torch.tensor([1, 2]),
        continuation=torch.ones(2, dtype=torch.bool),
        noise=noise,
    )
    torch.testing.assert_close(target.target_noise[:, :1], noise[:, :1])
    torch.testing.assert_close(
        target.target_noise[:, 1:], torch.full_like(target.target_noise[:, 1:], 3.0)
    )
    assert not torch.allclose(
        target.target_noise[:, 1:],
        torch.full_like(target.target_noise[:, 1:], gamma * 3.0),
    )


def test_evaluation_policy_action_defines_token_and_successor_condition() -> None:
    objective = make_objective()
    batch = make_batch(2)
    next_action = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
    target = objective.build_targets(
        None,
        batch,
        1,
        next_action=next_action,
        timesteps=torch.ones(2, dtype=torch.long),
        continuation=torch.zeros(2, dtype=torch.bool),
        noise=torch.zeros(2, 1, TOKEN_DIM),
    )
    torch.testing.assert_close(target.successor_conditioning[:, -ACT_DIM:], next_action)
    # Token layout is [r, s', a', flag]; the action occupies these coordinates.
    act_slice = slice(1 + OBS_DIM, 1 + OBS_DIM + ACT_DIM)
    # With zero noise, recover the clean token from z_t / alpha_t.
    alpha = objective.diffusion.sqrt_alphas_bar[1]
    recovered = target.z_t[:, 0] / alpha
    torch.testing.assert_close(recovered[:, act_slice], next_action)
    assert not torch.allclose(recovered[:, act_slice], batch.action)


def test_horizon_one_collapses_both_branches_to_stop() -> None:
    objective = make_objective(gamma=0.99)
    batch = make_batch(16)
    target = objective.build_targets(
        None,
        batch,
        1,
        next_action=torch.zeros(16, ACT_DIM),
        continuation=torch.ones(16, dtype=torch.bool),
    )
    assert not target.continuation.any()


def test_teacher_target_is_stop_gradient() -> None:
    objective = make_objective(steps=2)
    batch = make_batch(2)
    parameter = nn.Parameter(torch.tensor(2.0))

    def teacher(z_t, t, x, h):
        return parameter * torch.ones_like(z_t)

    target = objective.build_targets(
        teacher,
        batch,
        2,
        next_action=torch.zeros(2, ACT_DIM),
        continuation=torch.ones(2, dtype=torch.bool),
    )
    assert not target.target_noise.requires_grad


def test_forward_is_scalar_and_updates_only_student() -> None:
    objective = make_objective(steps=2)
    batch = make_batch(4)

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(TOKEN_DIM, TOKEN_DIM)

        def forward(self, z_t, t, x, h):
            return self.linear(z_t)

    student = Student()
    loss = objective(
        student,
        zero_teacher,
        batch,
        2,
        next_action=torch.zeros(4, ACT_DIM),
    )
    assert loss.ndim == 0 and torch.isfinite(loss) and loss >= 0
    loss.backward()
    assert student.linear.weight.grad is not None


def test_missing_teacher_for_continuation_raises() -> None:
    objective = make_objective()
    with pytest.raises(ValueError, match="teacher_noise_fn"):
        objective.build_targets(
            None,
            make_batch(1),
            2,
            next_action=torch.zeros(1, ACT_DIM),
            continuation=torch.ones(1, dtype=torch.bool),
        )


@pytest.mark.parametrize("mode", ["uniform", "snr", "min_snr_gamma"])
def test_loss_weights_are_finite(mode: str) -> None:
    schedule = DDPMSchedule.make_cosine(10)
    weight = get_loss_weight(torch.tensor([1, 5, 10]), schedule, mode=mode)
    assert weight.shape == (3,)
    assert torch.isfinite(weight).all()
    assert (weight >= 0).all()


def test_unknown_loss_weight_raises() -> None:
    with pytest.raises(ValueError, match="unknown loss weighting mode"):
        get_loss_weight(
            torch.tensor([1]), DDPMSchedule.make_cosine(2), mode="not-a-mode"
        )
