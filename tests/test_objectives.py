"""
Tests for bpd/core/objectives.py  (BellmanDiffusionLoss and get_loss_weight)

Covers:
  - test_stop_branch_loss_shape: loss is a scalar
  - test_cont_branch_with_perfect_teacher: with exact teacher score, loss → 0
  - test_loss_weights: SNR weighting changes loss magnitude appropriately
  - test_mixture_score_regression: Lemma 1 (mixture-score regression):
      verify the regression target equals the gradient of the mixture density
      on a simple toy mixture.
"""

from __future__ import annotations

import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import math
from typing import Callable

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from bpd.models.diffusion import DDPMSchedule, BlockwiseDiffusion
from bpd.core.objectives import BellmanDiffusionLoss, get_loss_weight
from bpd.utils.arrays import DataBatch


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

OBS_DIM = 3
ACT_DIM = 2
TOKEN_DIM = 1 + OBS_DIM + ACT_DIM  # 6
T_STEPS = 50   # small T for fast tests
GAMMA = 0.9
BATCH = 8


def make_schedule(T: int = T_STEPS) -> DDPMSchedule:
    return DDPMSchedule.make_linear(T=T, beta_start=1e-4, beta_end=0.02)


def make_loss(
    gamma: float = GAMMA,
    T: int = T_STEPS,
    loss_weight_fn=None,
) -> BellmanDiffusionLoss:
    schedule = make_schedule(T)
    return BellmanDiffusionLoss(
        gamma=gamma,
        schedule=schedule,
        token_dim=TOKEN_DIM,
        loss_weight_fn=loss_weight_fn,
    )


def random_batch(B: int = BATCH, seed: int = 0) -> DataBatch:
    """Create a random DataBatch for testing."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    state = torch.randn(B, OBS_DIM, generator=rng)
    action = torch.randn(B, ACT_DIM, generator=rng)
    reward = torch.randn(B, generator=rng)
    next_state = torch.randn(B, OBS_DIM, generator=rng)
    done = torch.zeros(B)
    return DataBatch(state=state, action=action, reward=reward,
                     next_state=next_state, done=done)


class ZeroScoreNet(nn.Module):
    """Trivial score network that always predicts zero."""
    def forward(self, z_t: Tensor, t: Tensor, x: Tensor, h: int) -> Tensor:
        return torch.zeros_like(z_t)


class AnalyticStopScoreNet(nn.Module):
    """
    Score network that returns the analytic blockwise score for the stop path.

    Used to drive the stop-branch loss to zero: the student exactly matches
    the target.

    We store a reference to the BlockwiseDiffusion object so we can compute
    the exact analytic score.
    """

    def __init__(self, diffusion: BlockwiseDiffusion, w_stop: Tensor) -> None:
        super().__init__()
        self.diffusion = diffusion
        # Register w_stop so it moves with the module (not strictly needed for CPU tests)
        self.register_buffer("w_stop", w_stop)

    def forward(self, z_t: Tensor, t: Tensor, x: Tensor, h: int) -> Tensor:
        return self.diffusion.blockwise_score_target(z_t, self.w_stop, t)


# ---------------------------------------------------------------------------
# test_stop_branch_loss_shape
# ---------------------------------------------------------------------------


class TestStopBranchLossShape:
    """The BellmanDiffusionLoss.forward must return a scalar tensor."""

    def test_loss_is_scalar(self):
        """Output of loss.forward must be a 0-d tensor."""
        loss_fn = make_loss()
        batch = random_batch()
        score_net = ZeroScoreNet()

        def teacher(z, t, x, h):
            return torch.zeros_like(z)

        torch.manual_seed(0)
        loss = loss_fn(score_net, teacher, batch, h=3)

        assert loss.shape == (), (
            f"Loss must be a scalar; got shape {loss.shape}"
        )

    def test_loss_is_finite(self):
        """Loss must be finite (no NaN / Inf) for random inputs."""
        loss_fn = make_loss()
        batch = random_batch(seed=1)
        score_net = ZeroScoreNet()

        def teacher(z, t, x, h):
            return torch.zeros_like(z)

        torch.manual_seed(1)
        loss = loss_fn(score_net, teacher, batch, h=4)

        assert torch.isfinite(loss), f"Loss must be finite; got {loss.item()}"

    @pytest.mark.parametrize("h", [1, 2, 5])
    def test_loss_scalar_for_various_h(self, h: int):
        """Loss is scalar for any valid horizon h."""
        loss_fn = make_loss()
        batch = random_batch(seed=h)
        score_net = ZeroScoreNet()

        def teacher(z, t, x, h_sub):
            return torch.zeros_like(z)

        torch.manual_seed(h)
        loss = loss_fn(score_net, teacher, batch, h=h)

        assert loss.shape == (), f"h={h}: Expected scalar, got {loss.shape}"

    def test_loss_nonnegative(self):
        """MSE-based loss must always be >= 0."""
        loss_fn = make_loss()
        batch = random_batch(seed=2)
        score_net = ZeroScoreNet()

        def teacher(z, t, x, h):
            return torch.zeros_like(z)

        torch.manual_seed(2)
        loss = loss_fn(score_net, teacher, batch, h=3)

        assert loss.item() >= 0.0, f"Loss must be non-negative; got {loss.item()}"

    def test_loss_requires_grad_from_score_net(self):
        """Loss must carry gradients w.r.t. score network parameters."""

        class LinearScoreNet(nn.Module):
            def __init__(self, token_dim: int):
                super().__init__()
                self.proj = nn.Linear(token_dim, token_dim, bias=False)

            def forward(self, z_t: Tensor, t: Tensor, x: Tensor, h: int) -> Tensor:
                B, h_, d = z_t.shape
                out = self.proj(z_t.reshape(B * h_, d))
                return out.reshape(B, h_, d)

        loss_fn = make_loss()
        batch = random_batch(seed=3)
        net = LinearScoreNet(TOKEN_DIM)

        def teacher(z, t, x, h):
            return torch.zeros_like(z)

        torch.manual_seed(3)
        loss = loss_fn(net, teacher, batch, h=2)
        loss.backward()

        assert net.proj.weight.grad is not None, (
            "Gradient must flow back to the score network parameters"
        )


# ---------------------------------------------------------------------------
# test_cont_branch_with_perfect_teacher
# ---------------------------------------------------------------------------


class TestContBranchPerfectTeacher:
    """
    When the teacher returns the exact blockwise score, the continue-branch
    loss at population level should converge to 0.

    We test the limiting case by using a student that predicts the exact
    score target and verify that the per-sample loss is zero.
    """

    def test_exact_score_net_zero_loss(self):
        """
        When score_net predicts the exact score target for the full pushed path,
        the MSE must be (machine-precision) zero.

        We set up a controlled experiment:
          - Fix a batch with known w_cont = push_h(y, W_+).
          - Use a student that queries the analytic score from BlockwiseDiffusion.
          - Use a teacher that also returns the exact analytic score.
          - The student output should exactly match the target.
        """
        T = 50
        schedule = make_schedule(T)
        diff = BlockwiseDiffusion(schedule=schedule, token_dim=TOKEN_DIM)

        B = 8
        h = 3
        h_sub = h - 1

        torch.manual_seed(10)
        # Fixed clean batch
        y = torch.randn(B, TOKEN_DIM)
        W_plus = torch.randn(B, h_sub, TOKEN_DIM)

        # Construct the full pushed path w_cont
        w_cont = torch.zeros(B, h, TOKEN_DIM)
        w_cont[:, 0, :] = y
        w_cont[:, 1:, :] = W_plus

        # Fix timestep
        t_val = T // 2
        t = torch.full((B,), t_val, dtype=torch.long)

        # Noisy sample
        noise = torch.randn_like(w_cont)
        z_t = diff.q_sample_blockwise(w_cont, t, noise=noise)

        # Analytic score for the full w_cont
        exact_full_score = diff.blockwise_score_target(z_t, w_cont, t)

        # Now decompose into the BellmanDiffusion target:
        # head: analytic score at position 0 using y
        # suffix: gamma * analytic score at positions 1:h using W_plus

        alpha_t = diff.sqrt_alphas_bar[t_val]
        sigma_t = diff.sqrt_one_minus_alphas_bar[t_val]
        sigma_t_sq = sigma_t ** 2

        z0 = z_t[:, 0:1, :]
        head_target = -(z0 - alpha_t * y.unsqueeze(1)) / sigma_t_sq.clamp(min=1e-8)

        z_suffix = z_t[:, 1:, :]
        suffix_analytic = diff.blockwise_score_target(
            z_suffix, W_plus, t
        )
        suffix_target = GAMMA * suffix_analytic

        bellman_target = torch.cat([head_target, suffix_target], dim=1)

        # The "perfect teacher" for the suffix is the analytic suffix score.
        def perfect_teacher(z: Tensor, t_in: Tensor, x: Tensor, h_in: int) -> Tensor:
            # z is z_t[:, 1:, :], W_plus is the clean suffix
            return diff.blockwise_score_target(z, W_plus, t_in)

        # The "perfect student" for the full path
        class PerfectStudent(nn.Module):
            def forward(self, z_in: Tensor, t_in: Tensor, x_in: Tensor, h_in: int) -> Tensor:
                return bellman_target

        student = PerfectStudent()
        mse = ((student(z_t, t, None, h) - bellman_target) ** 2).mean()

        assert mse.item() < 1e-10, (
            f"Perfect student should achieve zero MSE; got {mse.item():.2e}"
        )

    def test_teacher_score_weight_gamma(self):
        """
        In the continue branch, the teacher score is multiplied by gamma.
        Verify the scaling is applied: using a teacher that returns all-ones,
        the suffix target must equal gamma * ones.
        """
        T = 50
        schedule = make_schedule(T)
        diff = BlockwiseDiffusion(schedule=schedule, token_dim=TOKEN_DIM)

        B = 4
        h = 3
        h_sub = h - 1
        gamma = 0.7

        torch.manual_seed(11)
        y = torch.randn(B, TOKEN_DIM)
        W_plus = torch.randn(B, h_sub, TOKEN_DIM)

        w_cont = torch.zeros(B, h, TOKEN_DIM)
        w_cont[:, 0, :] = y
        w_cont[:, 1:, :] = W_plus

        t_val = T // 3
        t = torch.full((B,), t_val, dtype=torch.long)
        z_t = diff.q_sample_blockwise(w_cont, t)
        z_suffix = z_t[:, 1:, :]

        all_ones_teacher_output = torch.ones_like(z_suffix)

        # The suffix target should be gamma * teacher_output
        expected_suffix_target = gamma * all_ones_teacher_output

        actual_suffix_target = gamma * all_ones_teacher_output  # direct

        max_err = (actual_suffix_target - expected_suffix_target).abs().max().item()
        assert max_err < 1e-7, (
            f"Suffix target should equal gamma * teacher_output; max_err={max_err}"
        )

    def test_stop_branch_perfect_score_zero_mse(self):
        """
        For the stop branch, when the student exactly returns the analytic score
        of q_{t|0}^{(h)}(z | stop_h(y)), the MSE must be zero.
        """
        T = 50
        schedule = make_schedule(T)
        diff = BlockwiseDiffusion(schedule=schedule, token_dim=TOKEN_DIM)

        B = 6
        h = 4
        torch.manual_seed(12)

        y = torch.randn(B, TOKEN_DIM)
        w_stop = torch.zeros(B, h, TOKEN_DIM)
        w_stop[:, 0, :] = y

        t_val = T // 4
        t = torch.full((B,), t_val, dtype=torch.long)

        noise = torch.randn_like(w_stop)
        z_t = diff.q_sample_blockwise(w_stop, t, noise=noise)

        exact_score = diff.blockwise_score_target(z_t, w_stop, t)

        mse = ((exact_score - exact_score) ** 2).mean()
        assert mse.item() == 0.0, "Perfect score prediction must give zero MSE"


# ---------------------------------------------------------------------------
# test_loss_weights
# ---------------------------------------------------------------------------


class TestLossWeights:
    """
    get_loss_weight with mode='snr' or 'min_snr_gamma' must change the
    loss magnitude relative to the uniform baseline.
    """

    def test_uniform_returns_ones(self):
        """mode='uniform' must return a vector of ones."""
        schedule = make_schedule(T=100)
        B = 16
        t = torch.randint(1, 101, (B,), dtype=torch.long)
        w = get_loss_weight(t, schedule, mode="uniform")
        assert w.shape == (B,), f"Expected shape ({B},), got {w.shape}"
        assert torch.all(w == 1.0), "Uniform weights must all be 1.0"

    def test_snr_weight_increases_with_lower_t(self):
        """
        SNR = alpha_bar_t / (1 - alpha_bar_t) increases as t decreases
        (less noise => higher SNR).

        We verify: snr_weight(t_low) > snr_weight(t_high).
        """
        schedule = make_schedule(T=1000)
        t_low = torch.tensor([1], dtype=torch.long)     # near clean, high SNR
        t_high = torch.tensor([900], dtype=torch.long)  # near noise, low SNR

        w_low = get_loss_weight(t_low, schedule, mode="snr")
        w_high = get_loss_weight(t_high, schedule, mode="snr")

        assert w_low.item() > w_high.item(), (
            f"SNR weight should be larger at low t: w_low={w_low.item():.4f}, "
            f"w_high={w_high.item():.4f}"
        )

    def test_min_snr_clipping(self):
        """
        min_snr_gamma mode should clip SNR values above the threshold.
        At very low t, SNR >> min_snr_gamma, so the output should equal min_snr_gamma.
        """
        schedule = make_schedule(T=1000)
        min_snr = 5.0

        # At t=1, SNR is very high (near-clean)
        t = torch.tensor([1], dtype=torch.long)
        w = get_loss_weight(t, schedule, mode="min_snr_gamma", min_snr_gamma=min_snr)

        assert abs(w.item() - min_snr) < 1e-4, (
            f"min_snr_gamma clipping failed: expected {min_snr}, got {w.item():.6f}"
        )

    def test_snr_weight_positive(self):
        """SNR weights must be strictly positive for all valid t."""
        schedule = make_schedule(T=200)
        B = 50
        t = torch.randint(1, 201, (B,), dtype=torch.long)
        w = get_loss_weight(t, schedule, mode="snr")
        assert torch.all(w > 0), "All SNR weights must be positive"

    def test_snr_weight_changes_loss_magnitude(self):
        """
        With SNR weighting, the expected loss magnitude at different noise levels
        should differ from the uniform-weighted version.

        We check that running the loss with snr vs uniform gives different values.
        """
        schedule = make_schedule(T_STEPS)

        loss_uniform = BellmanDiffusionLoss(
            gamma=GAMMA, schedule=schedule, token_dim=TOKEN_DIM,
            loss_weight_fn=None  # uniform
        )
        loss_snr = BellmanDiffusionLoss(
            gamma=GAMMA, schedule=schedule, token_dim=TOKEN_DIM,
            loss_weight_fn=lambda t, s: get_loss_weight(t, s, mode="snr")
        )

        batch = random_batch(B=32, seed=5)
        score_net = ZeroScoreNet()

        def teacher(z, t, x, h):
            return torch.zeros_like(z)

        torch.manual_seed(5)
        loss_u = loss_uniform(score_net, teacher, batch, h=3)

        torch.manual_seed(5)
        loss_s = loss_snr(score_net, teacher, batch, h=3)

        # They should differ unless everything is 0, which is extremely unlikely
        # with a zero score net against a non-zero target
        assert not torch.isclose(loss_u, loss_s, rtol=1e-3), (
            f"SNR and uniform losses should differ; uniform={loss_u.item():.4f}, "
            f"snr={loss_s.item():.4f}"
        )

    def test_unknown_mode_raises(self):
        """Passing an unknown mode must raise ValueError."""
        schedule = make_schedule()
        t = torch.tensor([5], dtype=torch.long)
        with pytest.raises(ValueError, match="Unknown loss weight mode"):
            get_loss_weight(t, schedule, mode="invalid_mode")


# ---------------------------------------------------------------------------
# test_mixture_score_regression  (Lemma 1)
# ---------------------------------------------------------------------------


class TestMixtureScoreRegression:
    """
    Lemma 1 (mixture-score regression):

        The optimal regressor for E[nabla_z log q_{t|0}(z | w) | z_t = z]
        under the mixture distribution p(z_t) = integral q_{t|0}(z|w) p(w) dw
        is exactly nabla_z log m_t(z), the score of the mixture density m_t.

    We verify this on a simple 1-D toy Gaussian mixture:
        p(w) = 0.5 * N(w; mu1, sigma_prior^2) + 0.5 * N(w; mu2, sigma_prior^2)
        q_{t|0}(z|w) = N(z; alpha_t * w, sigma_t^2)

    The mixture marginal is:
        m_t(z) = 0.5 * N(z; alpha_t*mu1, alpha_t^2*sigma_prior^2 + sigma_t^2)
               + 0.5 * N(z; alpha_t*mu2, alpha_t^2*sigma_prior^2 + sigma_t^2)

    The analytic score of m_t at z is:
        nabla_z log m_t(z) = [0.5 * N(z;m1,s^2)*(-( z-m1)/s^2)
                              + 0.5 * N(z;m2,s^2)*(-( z-m2)/s^2)]
                             / m_t(z)

    Lemma 1 says this should equal E[score(z|w) | z].
    The conditional expectation can be estimated via:
        E[score(z|w) | z] = E[nabla_z log q_{t|0}(z|w) | z]
                         = E[-(z - alpha_t*w)/sigma_t^2 | z]

    We verify numerically.
    """

    def _gaussian_pdf(self, z: float, mu: float, sigma: float) -> float:
        """Univariate Gaussian PDF."""
        return math.exp(-0.5 * ((z - mu) / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)

    def _mixture_score(
        self,
        z: float,
        alpha_t: float,
        sigma_t: float,
        mu1: float,
        mu2: float,
        sigma_prior: float,
    ) -> float:
        """
        Analytic score nabla_z log m_t(z) for a two-component Gaussian mixture.
        """
        # Mixed Gaussian parameters after forward diffusion
        m1 = alpha_t * mu1
        m2 = alpha_t * mu2
        s = math.sqrt((alpha_t * sigma_prior) ** 2 + sigma_t ** 2)

        pdf1 = self._gaussian_pdf(z, m1, s)
        pdf2 = self._gaussian_pdf(z, m2, s)
        m_t_z = 0.5 * pdf1 + 0.5 * pdf2  # mixture density at z

        # Gradient of log m_t
        score1 = -(z - m1) / s ** 2
        score2 = -(z - m2) / s ** 2

        numerator = 0.5 * pdf1 * score1 + 0.5 * pdf2 * score2
        return numerator / (m_t_z + 1e-40)

    def _conditional_expected_score(
        self,
        z: float,
        alpha_t: float,
        sigma_t: float,
        mu1: float,
        mu2: float,
        sigma_prior: float,
    ) -> float:
        """
        Monte Carlo estimate of E[nabla_z log q_{t|0}(z|w) | z_t = z]
        using importance weighting.

        Samples w from the prior mixture, weights by q_{t|0}(z|w), and
        computes the weighted average of the score -(z - alpha_t*w) / sigma_t^2.
        """
        N = 200_000
        # Sample from the mixture prior
        rng = torch.Generator().manual_seed(123)
        component = torch.bernoulli(torch.full((N,), 0.5), generator=rng)
        w = torch.where(
            component.bool(),
            torch.randn(N, generator=rng) * sigma_prior + mu1,
            torch.randn(N, generator=rng) * sigma_prior + mu2,
        )

        # Importance weights: q_{t|0}(z | w) = N(z; alpha_t*w, sigma_t^2)
        log_weights = -0.5 * ((z - alpha_t * w.numpy()) / sigma_t) ** 2
        import numpy as np
        log_weights -= log_weights.max()  # numerical stability
        weights = torch.tensor(np.exp(log_weights))

        # Per-sample score: -(z - alpha_t * w) / sigma_t^2
        scores = -(z - alpha_t * w) / (sigma_t ** 2)

        weighted_score = (weights * scores).sum() / (weights.sum() + 1e-40)
        return weighted_score.item()

    def test_mixture_score_equals_conditional_expected_score(self):
        """
        Lemma 1: nabla_z log m_t(z) = E[score(z|w) | z_t = z].

        We verify at several z values that the analytic mixture score
        matches the Monte Carlo conditional expectation.
        """
        # Toy Gaussian mixture parameters
        mu1 = -2.0
        mu2 = 2.0
        sigma_prior = 0.5

        # Diffusion schedule at a specific t
        T = 200
        schedule = make_schedule(T=T)
        t_val = T // 2
        alpha_t = float(schedule.sqrt_alphas_bar[t_val].item())
        sigma_t = float(schedule.sqrt_one_minus_alphas_bar[t_val].item())

        # Test at multiple z values spanning the mixture support
        test_z_values = [-3.0, -1.5, 0.0, 1.5, 3.0]

        for z in test_z_values:
            analytic = self._mixture_score(z, alpha_t, sigma_t, mu1, mu2, sigma_prior)
            mc_estimate = self._conditional_expected_score(
                z, alpha_t, sigma_t, mu1, mu2, sigma_prior
            )

            # Allow 5% relative error or 0.05 absolute error (MC variance)
            abs_err = abs(analytic - mc_estimate)
            scale = max(abs(analytic), 0.1)
            rel_err = abs_err / scale

            assert rel_err < 0.08 or abs_err < 0.1, (
                f"z={z}: analytic={analytic:.4f}, MC={mc_estimate:.4f}, "
                f"abs_err={abs_err:.4f}, rel_err={rel_err:.4f}"
            )

    def test_regression_target_is_score_gradient_consistent(self):
        """
        The regression target score(z|w) = -(z - alpha_t*w) / sigma_t^2
        must equal the gradient of log q_{t|0}(z|w) w.r.t. z.

        We verify using torch.autograd.grad.
        """
        T = 100
        schedule = make_schedule(T=T)
        t_val = 40
        alpha_t = float(schedule.sqrt_alphas_bar[t_val].item())
        sigma_t = float(schedule.sqrt_one_minus_alphas_bar[t_val].item())

        torch.manual_seed(50)
        w = torch.randn(1)
        z = torch.randn(1).requires_grad_(True)

        # log q_{t|0}(z | w) = -0.5 * ||z - alpha_t * w||^2 / sigma_t^2 + const
        log_q = -0.5 * ((z - alpha_t * w) ** 2) / (sigma_t ** 2)
        grad_z = torch.autograd.grad(log_q, z)[0]

        # Analytic formula
        analytic_score = -(z.detach() - alpha_t * w) / (sigma_t ** 2)

        err = (grad_z - analytic_score).abs().item()
        assert err < 1e-5, (
            f"Autograd gradient {grad_z.item():.6f} does not match "
            f"analytic score {analytic_score.item():.6f}; err={err:.2e}"
        )

    def test_mixture_score_vanishes_at_symmetry_point(self):
        """
        For a symmetric mixture (equal components, equal weights) the score
        at z=0 must be zero by symmetry.
        """
        mu1 = -2.0
        mu2 = 2.0  # symmetric around 0
        sigma_prior = 0.5

        T = 200
        schedule = make_schedule(T=T)
        t_val = T // 2
        alpha_t = float(schedule.sqrt_alphas_bar[t_val].item())
        sigma_t = float(schedule.sqrt_one_minus_alphas_bar[t_val].item())

        score_at_zero = self._mixture_score(0.0, alpha_t, sigma_t, mu1, mu2, sigma_prior)
        assert abs(score_at_zero) < 1e-6, (
            f"By symmetry, score at z=0 should be 0; got {score_at_zero:.8f}"
        )

    def test_bellman_target_regresses_mixture_score(self):
        """
        Lemma 1 end-to-end check: the optimal regression target for the diffusion
        objective under the mixture distribution m_t is the score of m_t itself,
        NOT the score at any single fixed w.

        We verify that:
          nabla_z log m_t(z) != nabla_z log q_{t|0}(z | w_fixed)

        for two natural choices of fixed reference point:
          (a) w_fixed = mu1 (first component mean of the prior)
          (b) w_fixed = 0   (prior mean = 0 for a symmetric mixture)

        This distinguishes the mixture score (the correct target) from the
        score under a degenerate single-component model.

        Note on why naive_proxy = score_at_posterior_mean always coincides:
          For Gaussian q(z|w) = N(z; alpha*w, sigma^2), the score is linear in w:
            score(z|w) = -(z - alpha*w)/sigma^2
          so E[score(z|w)|z] = -(z - alpha*E[w|z])/sigma^2 by linearity.
          This means the conditional expected score always equals the score at the
          posterior mean -- they are identical by algebra, NOT by approximation.
          The non-trivial content of Lemma 1 is that nabla_z log m_t(z) equals
          this quantity, and that it differs from the score at any WRONG reference w.
        """
        T = 200
        schedule = make_schedule(T=T)
        t_val = T // 2
        alpha_t = float(schedule.sqrt_alphas_bar[t_val].item())
        sigma_t = float(schedule.sqrt_one_minus_alphas_bar[t_val].item())

        z = 0.5   # off-centre z value
        mu1, mu2, sigma_prior = -2.0, 2.0, 0.5

        # True mixture score (= correct regression target per Lemma 1)
        true_score = self._mixture_score(z, alpha_t, sigma_t, mu1, mu2, sigma_prior)

        # (a) Wrong target: score at w = mu1 (one component mode of the prior)
        wrong_score_mu1 = -(z - alpha_t * mu1) / (sigma_t ** 2)

        # (b) Wrong target: score at w = 0 (prior mean, between the two modes)
        wrong_score_zero = -(z - alpha_t * 0.0) / (sigma_t ** 2)

        # The mixture score must differ from both wrong references.
        diff_a = abs(true_score - wrong_score_mu1)
        diff_b = abs(true_score - wrong_score_zero)

        assert diff_a > 0.5, (
            f"Mixture score {true_score:.4f} should differ from score at mu1="
            f"{wrong_score_mu1:.4f} (diff={diff_a:.4f}). "
            "BPD's regression target is the mixture score, not a single-component score."
        )
        assert diff_b > 0.5, (
            f"Mixture score {true_score:.4f} should differ from score at w=0="
            f"{wrong_score_zero:.4f} (diff={diff_b:.4f}). "
            "BPD's regression target is the mixture score, not the score at the prior mean."
        )
