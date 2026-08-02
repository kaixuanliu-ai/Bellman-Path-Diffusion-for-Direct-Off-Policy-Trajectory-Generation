"""
Off-Policy Evaluation (OPE) estimator for Bellman Path Diffusion (BPD).

Implements Algorithm 2 (Direct Evaluation-Policy Trajectory Generation) and the
OPE estimator of Eq. 59 from:
  "Bellman Path Diffusion for Direct Off-Policy Trajectory Generation"

Key results implemented:

  Algorithm 2 – Direct Evaluation-Policy Trajectory Generation (Section 8.2):
    1. Sample s_0 ~ μ_0,  a_0 ~ π_e(·|s_0)
    2. Sample z_T ~ N(0, I_{Hd})
    3. Solve the reverse SDE from T to 0 conditioned on x_0 = (s_0, a_0)
    4. Decode  w_H = Φ_H^{-1}(z_0)
    5. Remove all token blocks after the first padding token

  OPE Estimator (Eq. 59):
    Ĵ_H(π) = (1/M) Σ_{m=1}^M Σ_{t=0}^{L_H(W_H^{(m)})-1} R_t^{(m)}

    NOTE: no explicit γ^t discount factor appears in the sum — geometric
    survival of real tokens implements the discounting by construction
    (Proposition 1, Eq. 15).

  Approximate OPE error bound (Corollary 1, Eq. 61):
    |J̄_H(π) - J_H(π)| ≤ δ_H^G ≤ Σ_{h=1}^H γ^{H-h} ε_h^G

  Length distribution law (Eq. 14):
    Pr(L_h = l) = (1 - γ) γ^{l-1}   for 1 ≤ l < h
    Pr(L_h = h) = γ^{h-1}

Shapes used throughout (consistent with the rest of the codebase):
    B          – batch size
    H          – maximum path horizon
    token_dim  – d_tok = 2 + obs_dim + act_dim (reward + obs + act + flag)
    T          – total diffusion timesteps
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from bpd.core.path import PAD_THRESHOLD
from bpd.models.diffusion import BlockwiseDiffusion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _geometric_length_pmf(h: int, gamma: float) -> np.ndarray:
    """Compute the theoretical geometric length PMF over {1, ..., h} (Eq. 14).

    Pr(L_h = l) = (1 - γ) γ^{l-1}   for 1 ≤ l < h
    Pr(L_h = h) = γ^{h-1}

    Args:
        h:     Path horizon.
        gamma: Discount factor γ ∈ (0, 1).

    Returns:
        1-D numpy array of shape (h,) indexed from l=1 to l=h.
    """
    pmf = np.empty(h, dtype=np.float64)
    for length in range(1, h):
        pmf[length - 1] = (1.0 - gamma) * gamma ** (length - 1)
    pmf[h - 1] = gamma ** (h - 1)
    return pmf


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL divergence KL(p || q) in nats.

    Only computes contributions where p > 0.  Uses eps to guard log(q).

    Args:
        p:   Empirical distribution, shape (n,).  Should sum to ~1.
        q:   Theoretical distribution, shape (n,).  Should sum to ~1.
        eps: Small constant added to q before taking logs.

    Returns:
        KL(p || q) as a Python float.
    """
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + eps))))


# ---------------------------------------------------------------------------
# OPEEvaluator
# ---------------------------------------------------------------------------


class OPEEvaluator:
    """Off-Policy Evaluator based on BPD direct trajectory generation.

    Implements the estimator from Section 8.2 of the BPD paper:

      * Algorithm 2: for each Monte-Carlo sample, draw (s_0, a_0) under the
        evaluation policy, sample z_T ~ N(0, I_{Hd}), run the DDPM reverse
        chain conditioned on x_0 = (s_0, a_0), and decode the resulting path.
      * Equation 59: average the undiscounted reward sums over real tokens.
      * Corollary 1 / Equation 61: bound the approximation error in terms of
        per-horizon score-matching residuals ε_h^G.
      * Equation 14: compare the empirical length distribution against the
        theoretical geometric law.

    Args:
        diffusion:        A :class:`~bpd.models.diffusion.BlockwiseDiffusion`
                          instance that implements the forward/reverse DDPM
                          chain for BPD.
        score_net:        Callable satisfying the
                          :class:`~bpd.models.diffusion.ScoreNet` protocol:
                          ``score_net(z_t, x, t) -> noise_pred``.
        schedule:         Noise schedule object exposing ``.T`` (total number
                          of diffusion steps).  Any object with a ``.T``
                          integer attribute is accepted (the schedule buffers
                          on ``diffusion`` are used for actual computation).
        normalizer:       Optional normalizer (e.g.
                          :class:`~bpd.data.normalizer.LimitsNormalizer`).
                          If provided, its ``unnormalize`` method is called on
                          decoded observation/action slices before rewards are
                          summed.  Pass ``None`` to skip unnormalization.
        eval_policy_fn:   Callable ``(states: Tensor) -> Tensor`` that maps a
                          batch of initial states (B, obs_dim) to a batch of
                          actions (B, act_dim) sampled from the evaluation
                          policy π_e.  This is used to draw a_0 ~ π_e(·|s_0).
        config:           Hyperparameter dictionary.  Recognised keys:

                          * ``"horizon"`` (int, required): path horizon H.
                          * ``"obs_dim"`` (int, required): observation dim.
                          * ``"act_dim"`` (int, required): action dim.
                          * ``"gamma"`` (float, default 0.99): discount γ
                            used to compute the theoretical length distribution.
                          * ``"norm_threshold"`` (float, default 0.3):
                            distance threshold around the zero padding
                            embedding used by the practical decoder.

    Example::

        evaluator = OPEEvaluator(
            diffusion=diffusion,
            score_net=score_net,
            schedule=schedule,
            normalizer=None,
            eval_policy_fn=policy.act,
            config={
                "horizon": 16,
                "obs_dim": 17,
                "act_dim": 6,
                "gamma": 0.99,
            },
        )
        initial_states = torch.zeros(32, 17, device="cuda")
        estimate = evaluator.estimate_return(
            initial_states, n_trajectories=32, device="cuda"
        )
    """

    def __init__(
        self,
        diffusion: BlockwiseDiffusion,
        score_net: Callable[[Tensor, Tensor, Tensor], Tensor],
        schedule: object,
        normalizer: Optional[object],
        eval_policy_fn: Callable[[Tensor], Tensor],
        config: Dict,
    ) -> None:
        # --- Mandatory config keys ---
        if "horizon" not in config:
            raise ValueError("config must contain 'horizon'.")
        if "obs_dim" not in config:
            raise ValueError("config must contain 'obs_dim'.")
        if "act_dim" not in config:
            raise ValueError("config must contain 'act_dim'.")

        self.diffusion = diffusion
        self.score_net = score_net
        self.schedule = schedule
        self.normalizer = normalizer
        self.eval_policy_fn = eval_policy_fn

        self.H: int = int(config["horizon"])
        self.obs_dim: int = int(config["obs_dim"])
        self.act_dim: int = int(config["act_dim"])
        # d_tok = reward + next_obs + next_action + injective-φ flag.
        self.token_dim: int = 2 + self.obs_dim + self.act_dim

        self.gamma: float = float(config.get("gamma", 0.99))
        self.norm_threshold: float = float(config.get("norm_threshold", 0.3))
        if self.H < 1:
            raise ValueError("horizon must be positive")

        # Validate token_dim matches diffusion model.
        if self.diffusion.token_dim != self.token_dim:
            raise ValueError(
                f"diffusion.token_dim={self.diffusion.token_dim} does not match "
                f"2 + obs_dim + act_dim = {self.token_dim} from config."
            )

    # -----------------------------------------------------------------------
    # Algorithm 2: Direct Evaluation-Policy Trajectory Generation
    # -----------------------------------------------------------------------

    def generate_trajectories(
        self,
        initial_states: Tensor,
        n_trajectories: int,
        device: Union[torch.device, str],
    ) -> List[Tensor]:
        """Generate evaluation-policy trajectories via Algorithm 2.

        For each of the ``n_trajectories`` Monte-Carlo samples:

          1. Draw s_0 from ``initial_states`` with replacement.
          2. Draw a_0 ~ π_e(·|s_0) using ``eval_policy_fn``.
          3. Concatenate x_0 = (s_0, a_0) as the conditioning context.
          4. Sample z_T ~ N(0, I_{H * token_dim}).
          5. Run the DDPM reverse chain from T to 0 conditioned on x_0.
          6. Decode the result and remove tokens after the first padding.

        Args:
            initial_states: Tensor of shape (N_init, obs_dim) containing
                            possible initial states s_0 ~ μ_0.  Sampled with
                            replacement to fill ``n_trajectories``
                            parallel chains.
            n_trajectories: Number M of Monte-Carlo trajectories to generate.
            device:         Compute device for diffusion sampling.

        Returns:
            List of M tensors, each of shape (L_m, token_dim) where L_m is
            the number of real (non-padding) tokens in the m-th trajectory.
            An empty trajectory (all padding) is returned as shape (0, token_dim).
        """
        device = torch.device(device) if isinstance(device, str) else device

        # ---- Step 1: sample s_0 ~ μ_0 ----------------------------------------
        n_init = initial_states.shape[0]
        if n_init < 1:
            raise ValueError("initial_states must contain at least one sample")
        # Independent draws with replacement implement s_0 ~ mu_0.
        idx = torch.randint(0, n_init, (n_trajectories,), device=device)
        s0 = initial_states[idx].to(device=device, dtype=torch.float32)  # (M, obs_dim)

        # ---- Step 2: sample a_0 ~ π_e(·|s_0) ---
        with torch.no_grad():
            a0 = self.eval_policy_fn(s0)  # (M, act_dim)

        a0 = a0.to(device=device, dtype=torch.float32)

        # ---- Step 3: form conditioning context x_0 = (s_0, a_0) ---
        x0 = torch.cat([s0, a0], dim=-1)  # (M, obs_dim + act_dim)

        # ---- Steps 4-5: sample z_T ~ N(0, I) and run reverse DDPM chain ---
        # p_sample_loop starts from pure Gaussian noise and iterates t=T→1.
        with torch.no_grad():
            z0 = self.diffusion.p_sample_loop(
                score_net=self.score_net,
                x=x0,
                h=self.H,
                batch_size=n_trajectories,
                device=device,
            )  # (M, H, token_dim)

        # ---- Step 6: decode each trajectory and remove padding ---
        trajectories: List[Tensor] = []
        for m in range(n_trajectories):
            tokens, is_real = self.decode_trajectory(z0[m], h=self.H)
            # Retain only the leading real tokens (stop at first padding).
            real_count = int(is_real.sum().item())
            trajectories.append(tokens[:real_count])  # (L_m, token_dim)

        return trajectories

    # -----------------------------------------------------------------------
    # OPE estimator (Eq. 59)
    # -----------------------------------------------------------------------

    def _unnormalize_rewards(self, rewards: Tensor, device: torch.device) -> Tensor:
        if self.normalizer is None:
            return rewards
        rewards_np = rewards.detach().cpu().numpy()
        if hasattr(self.normalizer, "unnormalize_rew"):
            values = self.normalizer.unnormalize_rew(rewards_np)
        elif hasattr(self.normalizer, "unnormalize"):
            values = self.normalizer.unnormalize(
                rewards_np.reshape(-1, 1), key="rewards"
            )
        else:
            raise TypeError("normalizer cannot unnormalize rewards")
        return torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1)

    def estimate_return(
        self,
        initial_states: Tensor,
        n_trajectories: int,
        device: Union[torch.device, str],
    ) -> Tensor:
        """Estimate J_H(π) via the OPE estimator Ĵ_H(π) (Eq. 59).

        Ĵ_H(π) = (1/M) Σ_{m=1}^M Σ_{t=0}^{L_H(W_H^{(m)})-1} R_t^{(m)}

        No explicit γ^t factor is applied — the geometric truncation of real
        tokens in the generated paths encodes the discounting structure by
        construction (Proposition 1 / Eq. 15).

        Args:
            initial_states: Tensor of shape (N_init, obs_dim).
            n_trajectories: Number M of Monte-Carlo samples.
            device:         Compute device.

        Returns:
            Scalar tensor containing the estimated policy value Ĵ_H(π).
        """
        device = torch.device(device) if isinstance(device, str) else device

        trajectories = self.generate_trajectories(
            initial_states=initial_states,
            n_trajectories=n_trajectories,
            device=device,
        )

        total_return = 0.0
        for traj in trajectories:
            if traj.shape[0] == 0:
                # Empty trajectory: all padding, contributes zero reward.
                continue
            # Rewards are stored in the first element (index 0) of each token.
            rewards = traj[:, 0]  # (L_m,)
            # Unnormalize rewards if a normalizer is available.
            rewards = self._unnormalize_rewards(rewards, device)
            # Eq. 59: undiscounted sum (geometric survival handles discounting).
            total_return += float(rewards.sum().item())

        estimate = total_return / max(n_trajectories, 1)
        return torch.tensor(estimate, dtype=torch.float32, device=device)

    # -----------------------------------------------------------------------
    # Length distribution diagnostics
    # -----------------------------------------------------------------------

    def compute_length_distribution(
        self,
        trajectories: List[Tensor],
    ) -> Dict[str, object]:
        """Compare the empirical path-length distribution against Eq. 14.

        Computes empirical Pr(L_H = l) from the decoded trajectories and
        compares it with the theoretical geometric law:

          Pr(L_H = l) = (1 - γ) γ^{l-1}   for 1 ≤ l < H
          Pr(L_H = H) = γ^{H-1}

        Also returns the KL divergence KL(empirical || theoretical) as a
        diagnostic for how well the model has learned the target length law.

        Args:
            trajectories: List of M decoded trajectory tensors as returned by
                          :meth:`generate_trajectories`.  Each tensor has shape
                          (L_m, token_dim).

        Returns:
            Dictionary with keys:

            * ``"empirical"``   – numpy array of shape (H,), empirical PMF
                                  over lengths {1, ..., H}.
            * ``"theoretical"`` – numpy array of shape (H,), theoretical PMF
                                  from Eq. 14 using the configured γ.
            * ``"kl"``          – float, KL(empirical || theoretical) in nats.
            * ``"lengths"``     – list of observed lengths (Python ints).
            * ``"mean_empirical"`` – float, empirical mean length.
            * ``"mean_theoretical"`` – float, theoretical mean length.
        """
        M = len(trajectories)
        if M == 0:
            raise ValueError("trajectories list is empty.")

        lengths = [traj.shape[0] for traj in trajectories]
        # Clip to [1, H]: treat length 0 (all-padding) as length 1 for PMF.
        lengths_clipped = [max(1, min(length, self.H)) for length in lengths]

        # Empirical PMF over {1, ..., H}.
        counts = np.zeros(self.H, dtype=np.float64)
        for length in lengths_clipped:
            counts[length - 1] += 1.0
        empirical = counts / M  # (H,)

        # Theoretical PMF from Eq. 14.
        theoretical = _geometric_length_pmf(self.H, self.gamma)

        # KL divergence.
        kl = _kl_divergence(empirical, theoretical)

        # Summary statistics.
        lengths_arr = np.array(lengths_clipped, dtype=np.float64)
        mean_empirical = float(lengths_arr.mean())
        # Theoretical mean: E[L_H] = Σ_{l=1}^H l * Pr(L_H=l)
        l_values = np.arange(1, self.H + 1, dtype=np.float64)
        mean_theoretical = float((l_values * theoretical).sum())

        return {
            "empirical": empirical,
            "theoretical": theoretical,
            "kl": kl,
            "lengths": lengths_clipped,
            "mean_empirical": mean_empirical,
            "mean_theoretical": mean_theoretical,
        }

    # -----------------------------------------------------------------------
    # Token decoding
    # -----------------------------------------------------------------------

    def decode_trajectory(
        self,
        z: Tensor,
        h: int,
    ) -> Tuple[Tensor, Tensor]:
        """Decode a noisy diffusion output into clean tokens and a real-token mask.

        This implements step 4 of Algorithm 2:
          ``w_H = Φ_H^{-1}(z_0)``

        Since the token map Φ is the identity encoding (tokens are directly
        concatenated and zero-padded), the decoding step consists of:

          1. Treating ``z`` (already denoised to z_0 by the reverse chain) as
             the path tensor.
          2. Classifying a token as padding when it lies near the zero padding
             embedding in full token space.
          3. Identifying the contiguous prefix of real tokens (padding is
             structurally appended after the last real token per Eq. 4-5).

        A token at position j is classified as padding when
        ``torch.norm(z[j]) < norm_threshold``.  The reward coordinate alone is
        never used: zero reward is a valid real transition.

        This practical heuristic is calibrated by ``config["norm_threshold"]``.

        Args:
            z: Denoised path tensor of shape (h, token_dim) – the z_0 output
               from the reverse diffusion chain for a single sample.
            h: Path horizon (number of token slots); must equal ``z.shape[0]``.

        Returns:
            Tuple ``(tokens, is_real)`` where:

            * ``tokens``  – Tensor of shape (h, token_dim), the decoded tokens
                            (same as ``z`` after optional clamping).
            * ``is_real`` – Boolean tensor of shape (h,), ``True`` at positions
                            that are classified as real (non-padding) tokens.

        Note:
            The BPD path structure guarantees that padding tokens appear as a
            suffix (Eq. 4-5).  Therefore ``is_real`` is the *prefix* indicator:
            once a padding position is encountered, all subsequent positions are
            also padding.  The caller of :meth:`generate_trajectories` relies
            on this contract to compute trajectory lengths.
        """
        if z.dim() != 2:
            raise ValueError(
                f"z must be 2-D (h, token_dim); got shape {tuple(z.shape)}"
            )
        if z.shape[0] != h:
            raise ValueError(f"z.shape[0]={z.shape[0]} does not match h={h}")
        if z.shape[1] != self.token_dim:
            raise ValueError(
                f"z.shape[1]={z.shape[1]} does not match token_dim={self.token_dim}"
            )

        tokens = z  # (h, token_dim); already denoised

        # --- Exact flag-based decode (Φ_H^{-1}, paper.tex L384) ---
        # Every real token carries flag = +1 in its last coordinate; the padding
        # token φ(⊥) carries flag = -1.  A slot is padding iff its flag is
        # negative.  This is the exact inverse on clean tokens and replaces the
        # earlier ‖token‖ < threshold heuristic (which could misread a near-zero
        # real transition as padding).
        is_padding = tokens[:, -1] < PAD_THRESHOLD  # (h,)

        is_real = ~is_padding
        # Every Bellman path contains its current transition (Eq. 6): slot 0 is
        # real by construction regardless of the generated flag value.
        if h > 0:
            is_real[0] = True

        # Enforce prefix-real / suffix-padding invariant (Eq. 4-5): once a slot
        # is padding, all subsequent slots are padding.
        if h > 0:
            cumulative_real = torch.cumprod(is_real.to(dtype=torch.long), dim=0).bool()
            is_real = cumulative_real

        return tokens, is_real

    # -----------------------------------------------------------------------
    # Corollary 1 error bound (Eq. 61)
    # -----------------------------------------------------------------------

    def corollary1_error_bound(
        self,
        return_stage_errors: List[float],
    ) -> float:
        """Compute the Corollary 1 upper bound on the OPE approximation error.

        Eq. 61 / Eq. 68:
            |J̄_H(π) - J_H(π)| ≤ δ_H^G ≤ Σ_{h=1}^H γ^{H-h} ε_h^G

        Here ε_h^G is the **return-seminorm (G-seminorm) Bellman stage error**
        of Eq. 66 — NOT a score-matching / denoising residual:

            ε_h^G = sup_x | V̂_h(x)
                            - ∫ κ_π(dy|x) [ r(y) + γ V̂_{h-1}(next_x(y)) ] |,

        i.e. the one-step Bellman residual of the *learned* path law's expected
        undiscounted return V̂_h(x) = E_{W_h ~ M̂_h(·|x)}[G_h(W_h)].  It measures
        how far the generated horizon-h return is from applying one exact
        evaluation-policy Bellman backup to the horizon-(h-1) return, and must
        be estimated empirically from generated trajectories, not read off the
        training loss.

        Args:
            return_stage_errors: List of H floats [ε_1^G, ..., ε_H^G], one per
                                 horizon level, each the G-seminorm stage error
                                 defined above.

        Returns:
            Upper bound δ_H^G as a Python float.

        Raises:
            ValueError: If the length of ``return_stage_errors`` does not equal
                        the configured horizon H.
        """
        if len(return_stage_errors) != self.H:
            raise ValueError(
                f"return_stage_errors must have length H={self.H}; "
                f"got {len(return_stage_errors)}."
            )

        # δ_H^G = Σ_{h=1}^H γ^{H-h} ε_h^G
        bound = 0.0
        for h_idx, eps_h in enumerate(return_stage_errors):
            # h_idx = h - 1 (0-based), so h = h_idx + 1
            h = h_idx + 1
            weight = self.gamma ** (self.H - h)  # γ^{H-h}
            bound += weight * eps_h

        return bound

    # -----------------------------------------------------------------------
    # Convenience: full evaluation pipeline
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        initial_states: Tensor,
        n_trajectories: int,
        device: Union[torch.device, str],
        return_stage_errors: Optional[List[float]] = None,
        verbose: bool = False,
    ) -> Dict[str, object]:
        """Run the complete OPE pipeline and return a diagnostics dictionary.

        Combines :meth:`estimate_return`, :meth:`compute_length_distribution`,
        and optionally :meth:`corollary1_error_bound` into a single call.

        Args:
            initial_states:            Initial state tensor (N_init, obs_dim).
            n_trajectories:            Number M of Monte-Carlo samples.
            device:                    Compute device.
            return_stage_errors:       Optional list of H G-seminorm return
                                       stage errors ε_h^G (Eq. 66) for the
                                       Corollary 1 bound.  If None the bound
                                       is not computed.
            verbose:                   If True, log summary statistics via the
                                       module logger at INFO level.

        Returns:
            Dictionary with keys:

            * ``"estimate"``         – scalar tensor, Ĵ_H(π) (Eq. 59).
            * ``"length_dist"``      – dict returned by
                                       :meth:`compute_length_distribution`.
            * ``"error_bound"``      – float, δ_H^G from Corollary 1, or None
                                       if ``return_stage_errors`` is None.
            * ``"n_trajectories"``   – int, M used.
            * ``"horizon"``          – int, H.
            * ``"gamma"``            – float, γ.
        """
        device = torch.device(device) if isinstance(device, str) else device

        # Generate trajectories (Algorithm 2).
        trajectories = self.generate_trajectories(
            initial_states=initial_states,
            n_trajectories=n_trajectories,
            device=device,
        )

        # Return estimate (Eq. 59).
        total_return = 0.0
        for traj in trajectories:
            if traj.shape[0] == 0:
                continue
            rewards = traj[:, 0]
            rewards = self._unnormalize_rewards(rewards, device)
            total_return += float(rewards.sum().item())

        estimate = torch.tensor(
            total_return / max(n_trajectories, 1),
            dtype=torch.float32,
            device=device,
        )

        # Length distribution diagnostics.
        length_dist = self.compute_length_distribution(trajectories)

        # Optional Corollary 1 bound.
        error_bound: Optional[float] = None
        if return_stage_errors is not None:
            error_bound = self.corollary1_error_bound(return_stage_errors)

        if verbose:
            logger.info(
                "OPE estimate: %.4f | mean length: %.2f (theoretical: %.2f) | "
                "KL(empirical||theoretical): %.4f | error_bound: %s",
                estimate.item(),
                length_dist["mean_empirical"],
                length_dist["mean_theoretical"],
                length_dist["kl"],
                f"{error_bound:.4f}" if error_bound is not None else "N/A",
            )

        return {
            "estimate": estimate,
            "length_dist": length_dist,
            "error_bound": error_bound,
            "n_trajectories": n_trajectories,
            "horizon": self.H,
            "gamma": self.gamma,
        }
