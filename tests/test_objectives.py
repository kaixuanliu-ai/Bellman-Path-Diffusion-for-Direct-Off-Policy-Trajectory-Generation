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


def make_objective(
    gamma: float = 0.8, steps: int = 4, prediction_type: str = "v"
) -> BellmanDiffusionLoss:
    return BellmanDiffusionLoss(
        gamma=gamma,
        schedule=DDPMSchedule.make_cosine(steps),
        token_dim=TOKEN_DIM,
        prediction_type=prediction_type,
    )


def zero_teacher(z_t, t, x, h):
    return torch.zeros_like(z_t)


def test_stop_target_is_sampled_noise() -> None:
    # Under epsilon-prediction the stop target is exactly the sampled noise.
    objective = make_objective(prediction_type="epsilon")
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


def test_stop_target_is_v_under_v_prediction() -> None:
    # Under v-prediction (default) the stop target is v = alpha_t*eps - sigma_t*x0.
    objective = make_objective(prediction_type="v")
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
    # Reconstruct the clean path by inverting q_sample: x0 = (z_t - sigma*eps)/alpha.
    t = torch.tensor([1, 2, 3])
    diff = objective.diffusion
    alpha = diff.sqrt_alphas_bar[t].view(-1, 1, 1)
    sigma = diff.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
    clean = (target.z_t - sigma * noise) / alpha
    expected = diff.v_target(clean, noise, t)
    torch.testing.assert_close(target.target_noise, expected)


def test_continuation_is_head_noise_plus_unscaled_teacher_noise() -> None:
    # Epsilon-prediction: head target is the sampled noise; suffix target is the
    # frozen teacher output with NO gamma multiplier.
    gamma = 0.25
    objective = make_objective(gamma=gamma, steps=2, prediction_type="epsilon")
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


def test_continuation_suffix_is_unscaled_teacher_under_v_prediction() -> None:
    # v-prediction: suffix target is still the frozen teacher output with NO
    # gamma multiplier; the head target is the v of the known head transition.
    gamma = 0.25
    # steps=4 keeps timesteps away from the terminal (alpha_T=0), where the
    # clean-path reconstruction by division would be singular.
    objective = make_objective(gamma=gamma, steps=4, prediction_type="v")
    batch = make_batch(2)
    next_action = torch.randn(2, ACT_DIM)
    noise = torch.randn(2, 3, TOKEN_DIM)
    timesteps = torch.tensor([1, 2])

    def constant_teacher(z_t, t, x, h):
        return torch.full_like(z_t, 3.0)

    target = objective.build_targets(
        constant_teacher,
        batch,
        3,
        next_action=next_action,
        timesteps=timesteps,
        continuation=torch.ones(2, dtype=torch.bool),
        noise=noise,
    )
    # Suffix target == teacher output, unscaled by gamma.
    torch.testing.assert_close(
        target.target_noise[:, 1:], torch.full_like(target.target_noise[:, 1:], 3.0)
    )
    # Head target == v of the (known) head transition; reconstruct its clean
    # value by inverting q_sample on the head block.
    diff = objective.diffusion
    alpha = diff.sqrt_alphas_bar[timesteps].view(-1, 1, 1)
    sigma = diff.sqrt_one_minus_alphas_bar[timesteps].view(-1, 1, 1)
    clean_head = (target.z_t[:, :1] - sigma * noise[:, :1]) / alpha
    expected_head = diff.v_target(clean_head, noise[:, :1], timesteps)
    torch.testing.assert_close(target.target_noise[:, :1], expected_head)


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


def test_branch_importance_sampling_is_unbiased() -> None:
    """Oversampling the stop branch must not change the population objective.

    Eq. 48 weights the branches by (1-gamma) and gamma.  Drawing the branch with
    probability p != gamma and reweighting by (1-gamma)/(1-p) and gamma/p is an
    importance-sampling estimator of the SAME quantity, so the expected loss
    must match the natural (p = gamma) estimator.
    """
    gamma, steps, h = 0.9, 4, 3
    batch = make_batch(512)
    next_action = torch.randn(512, ACT_DIM)

    class Const(nn.Module):
        def forward(self, z_t, t, x, hh):
            return torch.zeros_like(z_t)

    student = Const()

    def zero_teacher_fn(z_t, t, x, hh):
        return torch.zeros_like(z_t)

    def mean_loss(p, seed):
        obj = BellmanDiffusionLoss(
            gamma=gamma,
            schedule=DDPMSchedule.make_cosine(steps),
            token_dim=TOKEN_DIM,
            branch_sample_p=p,
        )
        torch.manual_seed(seed)
        vals = [
            float(obj(student, zero_teacher_fn, batch, h, next_action=next_action))
            for _ in range(40)
        ]
        return sum(vals) / len(vals)

    natural = mean_loss(gamma, 0)          # p = gamma (default estimator)
    oversampled = mean_loss(0.5, 0)        # heavy stop-branch oversampling

    # Both estimate the same population objective; allow Monte-Carlo slack.
    assert abs(natural - oversampled) / natural < 0.15, (
        f"IS estimator biased: natural={natural:.4f} oversampled={oversampled:.4f}"
    )


def test_branch_importance_weights_reduce_to_identity_at_p_equals_gamma() -> None:
    obj = BellmanDiffusionLoss(
        gamma=0.9, schedule=DDPMSchedule.make_cosine(4), token_dim=TOKEN_DIM
    )
    assert obj.branch_sample_p == pytest.approx(0.9)
    assert obj._w_stop == pytest.approx(1.0)
    assert obj._w_cont == pytest.approx(1.0)


def test_branch_sample_p_oversamples_stop_branch() -> None:
    obj = BellmanDiffusionLoss(
        gamma=0.95, schedule=DDPMSchedule.make_cosine(4), token_dim=TOKEN_DIM,
        branch_sample_p=0.5,
    )
    batch = make_batch(4000)
    torch.manual_seed(0)
    t = obj.build_targets(
        zero_teacher, batch, 3, next_action=torch.zeros(4000, ACT_DIM)
    )
    frac_cont = float(t.continuation.float().mean())
    # ~50% continuation instead of the natural 95%.
    assert 0.45 < frac_cont < 0.55
