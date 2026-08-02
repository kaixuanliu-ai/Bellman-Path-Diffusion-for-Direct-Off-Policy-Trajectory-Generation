"""
Tests for bpd/core/path.py

Covers:
  - stop_map: shape, first token, padding mask
  - push_map: prepend token, drop last
  - path_length: count non-padding tokens
  - compute_return: sum rewards excluding padding
  - test_length_law: Pr(L_h=l) matches Eq. 14 geometric distribution
  - test_value_identity: E[sum R_t] = E_pi[sum gamma^t R_t] (Eq. 15)
"""

from __future__ import annotations

import sys
import os

# ---------------------------------------------------------------------------
# Path setup: allow running from repo root or tests/ directory.
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import math

import pytest
import torch

from bpd.core.path import (
    stop_map,
    push_map,
    path_length,
    compute_return,
    length_log_prob,
    sample_path_length,
    encode_token,
    token_dim,
    PAD_FLAG,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
ACT_DIM = 2
D_TOK = token_dim(OBS_DIM, ACT_DIM)  # 7


def random_token(seed: int = 0) -> torch.Tensor:
    """Return a deterministic random token of shape (D_TOK,)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randn(D_TOK, generator=rng)


# ---------------------------------------------------------------------------
# test_stop_map
# ---------------------------------------------------------------------------


class TestStopMap:
    """stop_h(y) = (y, ⊥, ..., ⊥) per Eq. 4."""

    def test_shape_w(self):
        """Path tensor has shape (h, d_tok)."""
        h = 5
        y = random_token(1)
        w, _ = stop_map(y, h)
        assert w.shape == (h, D_TOK), f"Expected ({h}, {D_TOK}), got {w.shape}"

    def test_shape_pad_mask(self):
        """Padding mask has shape (h,)."""
        h = 5
        y = random_token(2)
        _, pad_mask = stop_map(y, h)
        assert pad_mask.shape == (h,), f"Expected ({h},), got {pad_mask.shape}"

    def test_first_token_equals_y(self):
        """Position 0 must equal y exactly."""
        h = 6
        y = random_token(3)
        w, _ = stop_map(y, h)
        assert torch.allclose(w[0], y.float()), "w[0] should equal y"

    def test_padding_positions_are_padding_token(self):
        """Positions 1..h-1 must be φ(⊥) = (0,...,0, PAD_FLAG) (injective φ)."""
        h = 5
        y = random_token(4)
        w, _ = stop_map(y, h)
        # All non-flag coordinates are zero.
        assert torch.all(w[1:, :-1] == 0.0), "Padding non-flag coords must be zero"
        # The flag coordinate marks padding (PAD_FLAG = -1), making φ injective.
        assert torch.all(w[1:, -1] == PAD_FLAG), "Padding flag coord must be PAD_FLAG"
        # A padding token is distinct from the real token y (injectivity).
        assert not torch.allclose(w[1], y.float())

    def test_pad_mask_first_false(self):
        """Position 0 is a real token: pad_mask[0] must be False."""
        h = 4
        y = random_token(5)
        _, pad_mask = stop_map(y, h)
        assert not pad_mask[0].item(), "pad_mask[0] must be False (real token)"

    def test_pad_mask_rest_true(self):
        """Positions 1..h-1 are padding: pad_mask[1:] must all be True."""
        h = 4
        y = random_token(6)
        _, pad_mask = stop_map(y, h)
        assert torch.all(pad_mask[1:]).item(), "pad_mask[1:] must all be True"

    def test_h_equals_1(self):
        """Edge case h=1: single real token, empty padding region."""
        y = random_token(7)
        w, pad_mask = stop_map(y, h=1)
        assert w.shape == (1, D_TOK)
        assert pad_mask.shape == (1,)
        assert not pad_mask[0].item()
        assert torch.allclose(w[0], y.float())

    def test_path_length_of_stop_map_is_one(self):
        """stop_h always has exactly one real token."""
        for h in [1, 3, 7]:
            y = random_token(h)
            _, pad_mask = stop_map(y, h)
            assert path_length(pad_mask) == 1, f"Expected L=1 for h={h}"


# ---------------------------------------------------------------------------
# test_push_map
# ---------------------------------------------------------------------------


class TestPushMap:
    """push_h(y, w) = (y, w_0, ..., w_{h-2}) per Eq. 5."""

    def _make_prev_path(self, h_minus_1: int, seed: int = 99):
        """Create a valid (h-1, d_tok) path and mask with mixed real/pad tokens."""
        rng = torch.Generator()
        rng.manual_seed(seed)
        w_prev = torch.randn(h_minus_1, D_TOK, generator=rng)
        # Mark last token as padding (required by push_map semantics).
        pad_mask_prev = torch.zeros(h_minus_1, dtype=torch.bool)
        if h_minus_1 > 1:
            pad_mask_prev[-1] = True  # last entry is padding
        return w_prev, pad_mask_prev

    def test_output_shape(self):
        """push_map output (w, pad_mask) must have shape (h, d_tok) and (h,)."""
        h = 5
        y = random_token(10)
        w_prev, pm_prev = self._make_prev_path(h - 1)
        w, pad_mask = push_map(y, w_prev, pm_prev)
        assert w.shape == (h, D_TOK), f"Expected ({h}, {D_TOK}), got {w.shape}"
        assert pad_mask.shape == (h,), f"Expected ({h},), got {pad_mask.shape}"

    def test_first_token_is_y(self):
        """Position 0 of the pushed path must equal y."""
        y = random_token(11)
        w_prev, pm_prev = self._make_prev_path(4)
        w, _ = push_map(y, w_prev, pm_prev)
        assert torch.allclose(w[0], y.float()), "push_map should place y at position 0"

    def test_suffix_is_w_prev_leading_tokens(self):
        """Positions 1..h-1 must equal w_prev[0..h-2] (first h-1 rows of w_prev)."""
        h_minus_1 = 4
        y = random_token(12)
        w_prev, pm_prev = self._make_prev_path(h_minus_1)
        w, _ = push_map(y, w_prev, pm_prev)
        # w[1:] should equal w_prev[:h_minus_1] which is all of w_prev
        assert torch.allclose(w[1:], w_prev[:h_minus_1].float()), (
            "push_map should copy w_prev[0:h-1] into positions 1..h-1"
        )

    def test_pad_mask_first_false(self):
        """Position 0 (the new head token y) must not be masked."""
        y = random_token(13)
        w_prev, pm_prev = self._make_prev_path(3)
        _, pad_mask = push_map(y, w_prev, pm_prev)
        assert not pad_mask[0].item(), "Position 0 must be real (pad_mask[0]=False)"

    def test_pad_mask_propagated(self):
        """Padding mask at positions 1.. should match pm_prev[0..h-2]."""
        h_minus_1 = 4
        y = random_token(14)
        w_prev, pm_prev = self._make_prev_path(h_minus_1, seed=42)
        _, pad_mask = push_map(y, w_prev, pm_prev)
        expected_suffix_mask = pm_prev[:h_minus_1]  # all of pm_prev
        assert torch.equal(pad_mask[1:], expected_suffix_mask), (
            "Suffix of pad_mask should match pm_prev[0:h_minus_1]"
        )

    def test_push_increases_path_length(self):
        """push_map of a new real token onto a stop path increases L by 1."""
        h = 5
        y1 = random_token(20)
        y2 = random_token(21)
        # Build a stop path of horizon h-1
        w_stop, pm_stop = stop_map(y1, h - 1)
        l_before = path_length(pm_stop)  # should be 1

        # Push y2 in front
        w_pushed, pm_pushed = push_map(y2, w_stop, pm_stop)
        l_after = path_length(pm_pushed)
        assert l_after == l_before + 1, (
            f"Expected L={l_before + 1} after push, got {l_after}"
        )

    def test_h_equals_2(self):
        """Edge case: push onto a length-1 path produces a length-2 path."""
        y_head = random_token(30)
        y_tail = random_token(31)
        w_stop, pm_stop = stop_map(y_tail, h=1)
        w, pad_mask = push_map(y_head, w_stop, pm_stop)
        assert w.shape == (2, D_TOK)
        assert path_length(pad_mask) == 2


# ---------------------------------------------------------------------------
# test_path_length
# ---------------------------------------------------------------------------


class TestPathLength:
    """path_length(pad_mask) = number of False entries."""

    def test_all_real(self):
        """All False mask => length equals h."""
        h = 6
        pad_mask = torch.zeros(h, dtype=torch.bool)
        assert path_length(pad_mask) == h

    def test_all_padding(self):
        """All True mask => length is 0."""
        h = 6
        pad_mask = torch.ones(h, dtype=torch.bool)
        assert path_length(pad_mask) == 0

    def test_mixed(self):
        """Manually constructed mask with known number of real tokens."""
        # True=pad, False=real
        pad_mask = torch.tensor([False, False, True, False, True, True], dtype=torch.bool)
        assert path_length(pad_mask) == 3  # 3 False entries

    def test_single_true(self):
        """Single padding position => length h-1."""
        h = 4
        pad_mask = torch.tensor([False, False, True, False], dtype=torch.bool)
        assert path_length(pad_mask) == 3

    def test_length_1_path(self):
        """Scalar edge case h=1."""
        pad_mask = torch.tensor([False], dtype=torch.bool)
        assert path_length(pad_mask) == 1

    def test_stop_map_always_gives_length_1(self):
        """stop_map for any h should produce a path of length 1."""
        for h in range(1, 8):
            y = random_token(h + 100)
            _, pm = stop_map(y, h)
            assert path_length(pm) == 1, f"h={h}: expected L=1"

    def test_returns_int(self):
        """path_length must return a plain Python int."""
        pad_mask = torch.zeros(3, dtype=torch.bool)
        result = path_length(pad_mask)
        assert isinstance(result, int), f"Expected int, got {type(result)}"


# ---------------------------------------------------------------------------
# test_compute_return
# ---------------------------------------------------------------------------


class TestComputeReturn:
    """compute_return sums w[:, 0] only over real (non-padding) positions."""

    def test_all_real_tokens(self):
        """With no padding, return equals the sum of all reward entries."""
        h, d = 4, D_TOK
        w = torch.zeros(h, d)
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        w[:, 0] = rewards
        pad_mask = torch.zeros(h, dtype=torch.bool)
        ret = compute_return(w, pad_mask)
        assert torch.isclose(ret, rewards.sum()), f"Expected {rewards.sum()}, got {ret}"

    def test_padding_excluded(self):
        """Reward at a padded position must not contribute to the return."""
        h, d = 4, D_TOK
        w = torch.zeros(h, d)
        w[:, 0] = torch.tensor([10.0, 20.0, 30.0, 99.0])  # 99 is padding reward
        pad_mask = torch.tensor([False, False, False, True], dtype=torch.bool)
        ret = compute_return(w, pad_mask)
        expected = 10.0 + 20.0 + 30.0
        assert torch.isclose(ret, torch.tensor(expected)), (
            f"Expected {expected}, got {ret.item()}"
        )

    def test_all_padding_gives_zero(self):
        """All-padding path => zero return."""
        h, d = 5, D_TOK
        w = torch.randn(h, d)
        pad_mask = torch.ones(h, dtype=torch.bool)
        ret = compute_return(w, pad_mask)
        assert torch.isclose(ret, torch.tensor(0.0)), f"Expected 0.0, got {ret.item()}"

    def test_stop_map_return(self):
        """Return of stop_h(y) with known reward equals that reward."""
        reward_val = 3.14
        next_obs = torch.zeros(OBS_DIM)
        next_act = torch.zeros(ACT_DIM)
        y = encode_token(reward_val, next_obs, next_act)
        h = 6
        w, pad_mask = stop_map(y, h)
        ret = compute_return(w, pad_mask)
        assert torch.isclose(ret, torch.tensor(reward_val, dtype=torch.float32)), (
            f"Expected {reward_val}, got {ret.item()}"
        )

    def test_non_reward_dims_ignored(self):
        """Dimensions 1..d_tok-1 (obs/act) must not affect the return."""
        h, d = 3, D_TOK
        w_a = torch.zeros(h, d)
        w_b = torch.zeros(h, d)
        # Same reward column
        w_a[:, 0] = 1.0
        w_b[:, 0] = 1.0
        # Different obs/act columns
        w_a[:, 1:] = torch.randn(h, d - 1)
        w_b[:, 1:] = torch.randn(h, d - 1)
        pad_mask = torch.zeros(h, dtype=torch.bool)
        assert torch.isclose(compute_return(w_a, pad_mask), compute_return(w_b, pad_mask)), (
            "Return must depend only on the reward column"
        )

    def test_scalar_output(self):
        """compute_return must return a 0-d tensor."""
        h, d = 3, D_TOK
        w = torch.zeros(h, d)
        pad_mask = torch.zeros(h, dtype=torch.bool)
        ret = compute_return(w, pad_mask)
        assert ret.shape == (), f"Expected scalar shape (), got {ret.shape}"


# ---------------------------------------------------------------------------
# test_length_law  (Eq. 14)
# ---------------------------------------------------------------------------


class TestLengthLaw:
    """Empirically verify Pr(L_h = l) matches the truncated geometric (Eq. 14)."""

    @pytest.mark.parametrize("gamma,h", [
        (0.9, 5),
        (0.5, 4),
        (0.99, 3),
    ])
    def test_empirical_vs_analytic(self, gamma: float, h: int):
        """
        Draw N samples from sample_path_length and compare empirical frequencies
        to the analytic probabilities from length_log_prob (Eq. 14).

        Tolerance: chi-square style check using L1 distance on normalised counts.
        """
        N = 50_000
        torch.manual_seed(42)
        counts = [0] * (h + 1)  # 1-indexed, so use h+1 slots

        for _ in range(N):
            l = sample_path_length(h, gamma)
            assert 1 <= l <= h, f"Sampled length {l} outside [1, {h}]"
            counts[l] += 1

        # Analytic probabilities
        analytic = {}
        for l in range(1, h + 1):
            analytic[l] = math.exp(length_log_prob(l, h, gamma).item())

        # Check L1 closeness: sum |empirical - analytic| < threshold
        l1_err = 0.0
        for l in range(1, h + 1):
            empirical = counts[l] / N
            l1_err += abs(empirical - analytic[l])

        threshold = 0.05  # 5% L1 tolerance for N=50k
        assert l1_err < threshold, (
            f"L1 error {l1_err:.4f} exceeds {threshold} for gamma={gamma}, h={h}. "
            f"Empirical: {[counts[l]/N for l in range(1,h+1)]}, "
            f"Analytic: {[analytic[l] for l in range(1,h+1)]}"
        )

    def test_probabilities_sum_to_one(self):
        """Analytic probabilities from Eq. 14 must sum to 1."""
        for gamma in [0.5, 0.9, 0.99]:
            for h in [2, 5, 10]:
                total = sum(
                    math.exp(length_log_prob(l, h, gamma).item())
                    for l in range(1, h + 1)
                )
                assert abs(total - 1.0) < 1e-5, (
                    f"Probs do not sum to 1 for gamma={gamma}, h={h}: got {total}"
                )

    def test_boundary_l_equals_h(self):
        """Pr(L_h = h) = gamma^{h-1} per Eq. 14."""
        gamma = 0.9
        h = 4
        expected = gamma ** (h - 1)
        got = math.exp(length_log_prob(h, h, gamma).item())
        assert abs(got - expected) < 1e-6, (
            f"Pr(L_{h}={h}) expected {expected}, got {got}"
        )

    def test_boundary_l_equals_1(self):
        """Pr(L_h = 1) = (1 - gamma) per Eq. 14."""
        gamma = 0.7
        h = 5
        expected = 1.0 - gamma
        got = math.exp(length_log_prob(1, h, gamma).item())
        assert abs(got - expected) < 1e-6, (
            f"Pr(L_{h}=1) expected {expected}, got {got}"
        )


# ---------------------------------------------------------------------------
# test_value_identity  (Eq. 15)
# ---------------------------------------------------------------------------


class TestValueIdentity:
    """
    Verify E[sum_{t < L_h} R_t] = E_pi[sum_{t=0}^{h-1} gamma^t R_t | X_0]

    We construct a simple deterministic MDP:
      - State space: {0, 1, ..., h-1}
      - Policy: always move to state+1
      - Reward at step t: R_t = r[t] (fixed reward vector)

    Then we Monte-Carlo estimate both sides and verify they agree.
    """

    def _simulate_lhs(
        self,
        rewards: list[float],
        gamma: float,
        h: int,
        n_samples: int,
    ) -> float:
        """
        LHS: E[sum_{t < L_h} R_t] where L_h ~ Pr(L_h=l) from Eq. 14.

        The path length L is drawn from the truncated geometric, and we sum
        the first L rewards (undiscounted).
        """
        total = 0.0
        for _ in range(n_samples):
            l = sample_path_length(h, gamma)
            total += sum(rewards[:l])
        return total / n_samples

    def _simulate_rhs(
        self,
        rewards: list[float],
        gamma: float,
        h: int,
        n_samples: int,
    ) -> float:
        """
        RHS: E_pi[sum_{t=0}^{h-1} gamma^t R_t] with the *same* reward vector.

        Since the MDP is deterministic and conditioning on X_0 is fixed, this
        equals the deterministic discounted sum sum_{t=0}^{h-1} gamma^t * R_t.
        (The expectation over the deterministic policy is trivial.)
        """
        discounted = sum(gamma ** t * rewards[t] for t in range(h))
        # The expectation is just this deterministic value repeated n_samples times.
        return discounted

    @pytest.mark.parametrize("gamma,h", [
        (0.9, 5),
        (0.5, 4),
        (0.8, 6),
    ])
    def test_value_identity(self, gamma: float, h: int):
        """
        Eq. 15: E[sum_{t<L_h} R_t] = E_pi[sum_{t=0}^{h-1} gamma^t R_t | X_0].

        Note: the LHS sums R_t *undiscounted* but with a random cutoff L.
        The geometric cutoff provides the discounting implicitly.
        We verify numerically with N=100k samples.
        """
        torch.manual_seed(0)
        n_samples = 100_000

        # Fixed reward sequence (arbitrary non-trivial values)
        rewards = [1.0 + 0.5 * t for t in range(h)]

        lhs = self._simulate_lhs(rewards, gamma, h, n_samples)
        rhs = self._simulate_rhs(rewards, gamma, h, n_samples)

        # Allow 2% relative error (from Monte Carlo variance with N=100k)
        rel_err = abs(lhs - rhs) / (abs(rhs) + 1e-8)
        assert rel_err < 0.02, (
            f"Value identity violated: LHS={lhs:.4f}, RHS={rhs:.4f}, "
            f"rel_err={rel_err:.4f} for gamma={gamma}, h={h}"
        )

    def test_value_identity_with_constant_reward(self):
        """
        With constant reward R_t = 1 for all t, the RHS is (1 - gamma^h) / (1 - gamma)
        and the LHS must agree to within Monte Carlo tolerance.
        """
        gamma = 0.9
        h = 5
        rewards = [1.0] * h
        n_samples = 100_000

        torch.manual_seed(1)
        lhs = self._simulate_lhs(rewards, gamma, h, n_samples)

        # Exact closed-form RHS: geometric series
        rhs = (1.0 - gamma ** h) / (1.0 - gamma)

        rel_err = abs(lhs - rhs) / (abs(rhs) + 1e-8)
        assert rel_err < 0.02, (
            f"Constant reward: LHS={lhs:.4f}, RHS={rhs:.4f}, rel_err={rel_err:.4f}"
        )
