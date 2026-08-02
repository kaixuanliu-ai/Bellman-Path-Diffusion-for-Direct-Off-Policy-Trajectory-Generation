"""
Tests for bpd/models/diffusion.py  (BlockwiseDiffusion)

Covers:
  - test_q_sample_shape: noisy tensor has same shape as clean
  - test_q_sample_noise_level: at t=1 (≈0), output ≈ clean; at t=T, output ≈ pure noise
  - test_blockwise_independence: two token blocks have independent noise
  - test_score_target_gradient: analytic score target has correct sign (points toward clean)
  - test_source_endpoint: at t=T, output is approximately standard Gaussian
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

import pytest
import torch

from bpd.models.diffusion import DDPMSchedule, BlockwiseDiffusion


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

T_STEPS = 200      # Small T for fast tests
TOKEN_DIM = 7
H = 4              # path horizon (number of blocks)
BATCH = 16


def make_diffusion(T: int = T_STEPS) -> BlockwiseDiffusion:
    schedule = DDPMSchedule.make_linear(T=T, beta_start=1e-4, beta_end=0.02)
    return BlockwiseDiffusion(schedule=schedule, token_dim=TOKEN_DIM)


def make_cosine_diffusion(T: int = T_STEPS) -> BlockwiseDiffusion:
    schedule = DDPMSchedule.make_cosine(T=T)
    return BlockwiseDiffusion(schedule=schedule, token_dim=TOKEN_DIM)


def clean_paths(batch: int = BATCH, h: int = H, seed: int = 0) -> torch.Tensor:
    """Return a deterministic clean path tensor of shape (B, h, token_dim)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randn(batch, h, TOKEN_DIM, generator=rng)


# ---------------------------------------------------------------------------
# test_q_sample_shape
# ---------------------------------------------------------------------------


class TestQSampleShape:
    """q_sample_blockwise must return a tensor of the same shape as the input."""

    def test_same_shape_as_clean(self):
        diff = make_diffusion()
        w = clean_paths()
        t = torch.randint(1, T_STEPS + 1, (BATCH,), dtype=torch.long)
        z_t = diff.q_sample_blockwise(w, t)
        assert z_t.shape == w.shape, (
            f"Expected shape {w.shape}, got {z_t.shape}"
        )

    def test_shape_with_explicit_noise(self):
        """Providing explicit noise must not change the output shape."""
        diff = make_diffusion()
        w = clean_paths(batch=8, h=6)
        t = torch.ones(8, dtype=torch.long) * 50
        noise = torch.zeros_like(w)
        z_t = diff.q_sample_blockwise(w, t, noise=noise)
        assert z_t.shape == w.shape, (
            f"Expected shape {w.shape}, got {z_t.shape}"
        )

    @pytest.mark.parametrize("batch,h,d", [
        (1, 1, 7),
        (4, 3, 7),
        (32, 8, 7),
    ])
    def test_various_batch_h_configurations(self, batch: int, h: int, d: int):
        """Shape must be (B, h, TOKEN_DIM) across various B and h."""
        schedule = DDPMSchedule.make_linear(T=T_STEPS)
        diff = BlockwiseDiffusion(schedule=schedule, token_dim=d)
        w = torch.randn(batch, h, d)
        t = torch.randint(1, T_STEPS + 1, (batch,), dtype=torch.long)
        z_t = diff.q_sample_blockwise(w, t)
        assert z_t.shape == (batch, h, d), (
            f"Expected ({batch},{h},{d}), got {z_t.shape}"
        )


# ---------------------------------------------------------------------------
# test_q_sample_noise_level
# ---------------------------------------------------------------------------


class TestQSampleNoiseLevel:
    """
    Check boundary behaviour of the forward kernel:
      - At t=1 (smallest step, near clean): output is close to clean.
      - At t=T (maximum noise): output is far from clean and resembles noise.
    """

    def test_at_t1_output_close_to_clean(self):
        """
        At t=1, alpha_1 ≈ 1 and sigma_1 ≈ 0 (small beta_1 = beta_start).
        With zero noise, output should equal alpha_1 * w, which is close to w.
        """
        diff = make_diffusion(T=1000)
        w = clean_paths(batch=16)
        t = torch.ones(16, dtype=torch.long)  # t=1
        noise = torch.zeros_like(w)

        z_t = diff.q_sample_blockwise(w, t, noise=noise)

        # alpha_1 = sqrt(alpha_bar_1) ≈ sqrt(1 - beta_start) ≈ 1
        alpha_1 = float(diff.sqrt_alphas_bar[1].item())
        expected = alpha_1 * w

        max_diff = (z_t - expected).abs().max().item()
        assert max_diff < 1e-5, (
            f"At t=1 with zero noise, z_t should equal alpha_1*w; max_diff={max_diff}"
        )

    def test_at_tT_output_dominated_by_noise(self):
        """
        At t=T, alpha_T ≈ 0 and sigma_T ≈ 1.
        With unit noise, output should be very close to the noise itself
        (the clean signal is almost entirely suppressed).
        """
        T = 1000
        diff = make_diffusion(T=T)
        w = clean_paths(batch=32)
        t = torch.full((32,), T, dtype=torch.long)
        noise = torch.randn_like(w)

        z_t = diff.q_sample_blockwise(w, t, noise=noise)

        # sigma_T ≈ 1; the signal contribution alpha_T * w ≈ 0
        alpha_T = float(diff.sqrt_alphas_bar[T].item())
        sigma_T = float(diff.sqrt_one_minus_alphas_bar[T].item())

        # The maximum signal contamination is alpha_T * |w_max|
        w_abs_max = w.abs().max().item()
        signal_contamination = alpha_T * w_abs_max

        # All elements of z_t should be within (sigma_T * noise ± signal_contamination)
        expected_approx = sigma_T * noise
        max_diff = (z_t - expected_approx).abs().max().item()

        # Allow tolerance up to the signal contamination level (plus small float err)
        assert max_diff <= signal_contamination + 1e-4, (
            f"At t=T, output should be dominated by noise. "
            f"Max diff from sigma_T*noise: {max_diff:.6f}, "
            f"alpha_T={alpha_T:.6f}, signal_contamination={signal_contamination:.6f}"
        )

    def test_noise_level_increases_monotonically(self):
        """
        The expected squared distance E[||z_t - alpha_t * w||^2] = sigma_t^2 * D
        should increase monotonically with t (linear DDPM schedule).
        """
        T = 100
        schedule = DDPMSchedule.make_linear(T=T, beta_start=1e-4, beta_end=0.02)
        diff = BlockwiseDiffusion(schedule=schedule, token_dim=TOKEN_DIM)

        w = torch.zeros(1, H, TOKEN_DIM)  # zero clean path for clean measurement
        variance_levels = []

        for t_val in [1, T // 4, T // 2, 3 * T // 4, T]:
            t = torch.tensor([t_val], dtype=torch.long)
            # With w=0, z_t = sigma_t * epsilon => var = sigma_t^2
            # Use many noise samples to estimate
            rng = torch.Generator().manual_seed(t_val)
            samples = torch.stack([
                diff.q_sample_blockwise(w, t)
                for _ in range(200)
            ])
            empirical_var = samples.var().item()
            variance_levels.append(empirical_var)

        # Variance should be non-decreasing
        for i in range(len(variance_levels) - 1):
            assert variance_levels[i] <= variance_levels[i + 1] + 0.05, (
                f"Variance not monotone: levels={variance_levels}"
            )


# ---------------------------------------------------------------------------
# test_blockwise_independence
# ---------------------------------------------------------------------------


class TestBlockwiseIndependence:
    """
    Per Eq. 24, the blockwise kernel is a product distribution:
    blocks j and k must be independently noised.

    We verify this by checking that the noise injected into block 0 and
    block 1 are uncorrelated across many samples.
    """

    def test_noise_uncorrelated_across_blocks(self):
        """
        Fix clean path w=0. Then z_t[:, 0, :] = sigma_t * eps_0 and
        z_t[:, 1, :] = sigma_t * eps_1 with eps_0, eps_1 independent.

        The sample correlation between the first element of each block
        across N trials should be near 0.
        """
        T = 100
        diff = make_diffusion(T=T)
        N = 2000

        w = torch.zeros(N, H, TOKEN_DIM)
        t = torch.full((N,), T // 2, dtype=torch.long)

        torch.manual_seed(7)
        z_t = diff.q_sample_blockwise(w, t)

        # Extract a single dimension from two different blocks
        block0_dim0 = z_t[:, 0, 0]  # shape (N,)
        block1_dim0 = z_t[:, 1, 0]  # shape (N,)

        # Pearson correlation coefficient
        def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
            xc = x - x.mean()
            yc = y - y.mean()
            denom = (xc.std() * yc.std()).clamp(min=1e-8)
            return float((xc * yc).mean() / denom)

        corr = pearson(block0_dim0, block1_dim0)
        assert abs(corr) < 0.08, (
            f"Blocks 0 and 1 should have uncorrelated noise, got corr={corr:.4f}"
        )

    def test_noise_uncorrelated_across_token_dims_within_block(self):
        """
        Within a single block, independent dimensions of the noise must
        also be uncorrelated (Gaussian isotropy).
        """
        T = 100
        diff = make_diffusion(T=T)
        N = 2000

        w = torch.zeros(N, H, TOKEN_DIM)
        t = torch.full((N,), T // 2, dtype=torch.long)

        torch.manual_seed(8)
        z_t = diff.q_sample_blockwise(w, t)

        # Two different dims within the same block
        dim_a = z_t[:, 0, 0]
        dim_b = z_t[:, 0, 3]

        def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
            xc = x - x.mean()
            yc = y - y.mean()
            denom = (xc.std() * yc.std()).clamp(min=1e-8)
            return float((xc * yc).mean() / denom)

        corr = pearson(dim_a, dim_b)
        assert abs(corr) < 0.08, (
            f"Dims 0 and 3 within block 0 should be uncorrelated, got corr={corr:.4f}"
        )

    def test_different_t_samples_give_different_noise_levels(self):
        """
        Using t=1 vs t=T should produce samples with very different variances.
        """
        T = 200
        diff = make_diffusion(T=T)
        N = 500
        w = torch.zeros(N, H, TOKEN_DIM)

        torch.manual_seed(9)
        z_low = diff.q_sample_blockwise(w, torch.ones(N, dtype=torch.long))
        z_high = diff.q_sample_blockwise(w, torch.full((N,), T, dtype=torch.long))

        var_low = z_low.var().item()
        var_high = z_high.var().item()

        assert var_low < var_high, (
            f"t=1 should have lower variance than t=T: var_low={var_low:.4f}, "
            f"var_high={var_high:.4f}"
        )


# ---------------------------------------------------------------------------
# test_score_target_gradient
# ---------------------------------------------------------------------------


class TestScoreTargetGradient:
    """
    The analytic score target must point in the direction of the clean data:
        score = -(z - alpha_t * w) / sigma_t^2

    At any noisy z, the score should have the opposite sign of (z - alpha_t * w),
    i.e., it should point from z toward the mean alpha_t * w.
    """

    def test_score_points_toward_mean(self):
        """
        For each (b, j, d), sign(score[b,j,d]) should equal -sign(z[b,j,d] - alpha_t*w[b,j,d]).
        We verify this on a batch of non-degenerate (large signal, large noise) samples.
        """
        diff = make_diffusion()
        B = 32
        t_val = T_STEPS // 2  # mid-noise level

        torch.manual_seed(42)
        w = torch.randn(B, H, TOKEN_DIM) * 2.0  # amplify to avoid near-zero issues
        t = torch.full((B,), t_val, dtype=torch.long)

        noise = torch.randn(B, H, TOKEN_DIM) * 2.0
        z_t = diff.q_sample_blockwise(w, t, noise=noise)

        score = diff.blockwise_score_target(z_t, w, t)

        alpha_t = float(diff.sqrt_alphas_bar[t_val].item())
        residual = z_t - alpha_t * w  # shape (B, H, TOKEN_DIM)

        # Where |residual| > 0.05, the score should have opposite sign
        large = residual.abs() > 0.05
        sign_ok = (score[large] * residual[large] < 0)
        frac_correct = sign_ok.float().mean().item()

        assert frac_correct > 0.99, (
            f"Expected >99% of large-residual positions to have score pointing "
            f"toward mean; got {frac_correct:.4f}"
        )

    def test_score_magnitude(self):
        """
        score = -(z - alpha_t * w) / sigma_t^2
        With known noise and clean path, the score magnitude must match.
        """
        diff = make_diffusion()
        B = 8
        t_val = 50

        alpha_t = float(diff.sqrt_alphas_bar[t_val].item())
        sigma_t = float(diff.sqrt_one_minus_alphas_bar[t_val].item())

        w = torch.randn(B, H, TOKEN_DIM)
        noise = torch.randn(B, H, TOKEN_DIM)
        z_t = alpha_t * w + sigma_t * noise

        t = torch.full((B,), t_val, dtype=torch.long)
        score = diff.blockwise_score_target(z_t, w, t)

        # Analytic: score = -noise / sigma_t  (since z - alpha*w = sigma*noise)
        expected_score = -noise / sigma_t

        max_err = (score - expected_score).abs().max().item()
        assert max_err < 1e-4, (
            f"Score magnitude mismatch: max_err={max_err:.6f}"
        )

    def test_score_at_clean_data_is_zero_mean(self):
        """
        At z = alpha_t * w (zero noise), the score should be zero.
        """
        diff = make_diffusion()
        B = 16
        t_val = 100

        w = torch.randn(B, H, TOKEN_DIM)
        alpha_t = float(diff.sqrt_alphas_bar[t_val].item())
        z_mean = alpha_t * w  # zero residual

        t = torch.full((B,), t_val, dtype=torch.long)
        score = diff.blockwise_score_target(z_mean, w, t)

        max_abs = score.abs().max().item()
        assert max_abs < 1e-5, (
            f"Score at the mean should be zero, got max_abs={max_abs:.6f}"
        )

    def test_score_shape_matches_z(self):
        """blockwise_score_target must return a tensor of the same shape as z."""
        diff = make_diffusion()
        B = 6
        w = clean_paths(batch=B)
        t = torch.randint(1, T_STEPS + 1, (B,), dtype=torch.long)
        noise = torch.randn_like(w)
        z_t = diff.q_sample_blockwise(w, t, noise=noise)

        score = diff.blockwise_score_target(z_t, w, t)
        assert score.shape == z_t.shape, (
            f"Expected shape {z_t.shape}, got {score.shape}"
        )


# ---------------------------------------------------------------------------
# test_source_endpoint
# ---------------------------------------------------------------------------


class TestSourceEndpoint:
    """
    At t=T, the forward process should map any clean data to approximately
    N(0, I).  We verify this statistically.
    """

    def test_marginal_at_tT_is_approximately_standard_gaussian(self):
        """
        Eq. 25: m_{h,T}^pi ≈ N(0, I) (source distribution).

        With w drawn from a non-trivial distribution, z_T ~ q_{T|0}(.|w) should
        be approximately N(0, I) when alpha_T ≈ 0 and sigma_T ≈ 1.

        We check:
          - Mean ≈ 0  (within 3 standard errors)
          - Variance ≈ 1  (within 5%)
        """
        T = 1000
        diff = make_diffusion(T=T)
        N = 5000  # total samples

        torch.manual_seed(100)
        # Sample diverse clean paths (non-zero mean, large spread)
        w = torch.randn(N, H, TOKEN_DIM) * 3.0 + 2.0
        t = torch.full((N,), T, dtype=torch.long)

        z_T = diff.q_sample_blockwise(w, t)

        # Flatten to check marginal
        z_flat = z_T.reshape(-1).float()

        mean = z_flat.mean().item()
        std = z_flat.std().item()

        # Standard error of mean for N*H*TOKEN_DIM samples
        n_total = z_flat.numel()
        se_mean = 1.0 / math.sqrt(n_total)

        assert abs(mean) < 5 * se_mean, (
            f"Mean at t=T should be ~0; got {mean:.4f} (SE={se_mean:.4f})"
        )
        assert abs(std - 1.0) < 0.05, (
            f"Std at t=T should be ~1; got {std:.4f}"
        )

    def test_output_distribution_kolmogorov_smirnov(self):
        """
        A stricter test: run a KS test against N(0,1) for samples at t=T.
        We use scipy.stats.kstest and require p-value > 0.01.
        """
        pytest.importorskip("scipy")
        from scipy import stats

        T = 1000
        diff = make_diffusion(T=T)
        N = 2000

        torch.manual_seed(200)
        w = torch.randn(N, 1, TOKEN_DIM) * 2.0
        t = torch.full((N,), T, dtype=torch.long)

        z_T = diff.q_sample_blockwise(w, t)
        samples = z_T.reshape(-1).numpy()

        stat, p_value = stats.kstest(samples, "norm", args=(0.0, 1.0))
        assert p_value > 0.01, (
            f"KS test rejected N(0,1) at t=T: stat={stat:.4f}, p={p_value:.4f}. "
            "The source distribution may not be approximately standard Gaussian."
        )

    def test_at_t1_output_not_standard_gaussian(self):
        """
        Sanity check: at t=1 (near-clean), z_1 should NOT look like N(0,1)
        when w is far from zero.
        """
        T = 1000
        diff = make_diffusion(T=T)
        N = 1000

        torch.manual_seed(300)
        # Clean path with large positive mean so z_1 ≈ alpha_1 * w >> 0
        w = torch.ones(N, H, TOKEN_DIM) * 5.0
        t = torch.ones(N, dtype=torch.long)  # t=1

        z_1 = diff.q_sample_blockwise(w, t)
        mean_abs = z_1.mean().abs().item()

        # z_1 ≈ alpha_1 * 5 >> 0; if this were N(0,1) the mean would be ~0
        assert mean_abs > 1.0, (
            f"At t=1 with large positive clean signal, mean should be >>0; "
            f"got {mean_abs:.4f}"
        )


class TestVPredictionSourceStability:
    """v-prediction keeps x0 recovery finite at the exact source alpha_T=0."""

    def test_v_x0_recovery_finite_at_source(self):
        # With zero-terminal-SNR the terminal alpha_T is exactly 0.
        diff = BlockwiseDiffusion(
            DDPMSchedule.make_cosine(T=T_STEPS), token_dim=TOKEN_DIM,
            prediction_type="v",
        )
        assert float(diff.sqrt_alphas_bar[-1]) == 0.0
        z_T = torch.randn(BATCH, H, TOKEN_DIM)
        v_pred = torch.randn(BATCH, H, TOKEN_DIM)
        t = torch.full((BATCH,), T_STEPS, dtype=torch.long)  # terminal index
        x0 = diff.predict_x0(z_T, t, v_pred)
        # v-recovery x0 = alpha_T*z_T - sigma_T*v = -v at the source: finite.
        assert torch.isfinite(x0).all()
        torch.testing.assert_close(x0, -v_pred)

    def test_epsilon_x0_recovery_is_singular_at_source(self):
        # The epsilon recovery divides by alpha_T (clamped) -> huge amplification,
        # which is exactly why v-prediction is used for the exact source.
        diff = BlockwiseDiffusion(
            DDPMSchedule.make_cosine(T=T_STEPS), token_dim=TOKEN_DIM,
            prediction_type="epsilon",
        )
        z_T = torch.randn(BATCH, H, TOKEN_DIM)
        eps_pred = torch.zeros(BATCH, H, TOKEN_DIM)  # eps != z_T -> no cancellation
        t = torch.full((BATCH,), T_STEPS, dtype=torch.long)
        x0 = diff.predict_x0(z_T, t, eps_pred)
        # (z_T - 0)/clamp(0,1e-8) = 1e8 * z_T -> enormous magnitude.
        assert x0.abs().max().item() > 1e6

    def test_v_reverse_chain_is_finite(self):
        diff = BlockwiseDiffusion(
            DDPMSchedule.make_cosine(T=50), token_dim=6, prediction_type="v",
        )

        def net(z, x, t):
            return torch.zeros_like(z)

        out = diff.p_sample_loop(net, x=torch.zeros(4, 3), h=3, batch_size=4, device="cpu")
        assert torch.isfinite(out).all()
