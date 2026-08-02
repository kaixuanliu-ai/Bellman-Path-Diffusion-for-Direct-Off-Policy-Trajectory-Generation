"""
Bellman branches for Bellman Path Diffusion (BPD).

Implements the two Bellman branches of Algorithm 1 (Training, lines 9-22) and
Section 5 of "Bellman Path Diffusion for Direct Off-Policy Trajectory Generation".

Overview
--------
BPD trains a score network s_theta(z, t | x, h) to match the score of the
*h-step noisy path distribution* m_{h,t}^pi.  This score is obtained via a
Bellman recursion with two branches (Section 5.2):

Stop branch (Eq. 41-43)
    Handles the event that the sampled path ends at depth h (path length = 1).
    The target path is stop_h(y) = (y, pad, ..., pad), which places the current
    token y at position 0 and fills the remainder with padding (zeros).
    The analytic score of the blockwise Gaussian kernel factorises cleanly:

        U_{h,t}^stop = grad_z log q_{t|0}^{(h)}(Z^stop | stop_h(y))
                     = -1/sigma_t^2 * [Z^stop - alpha_t * Phi_h(stop_h(y))]

    where Phi_h maps each token to its embedding (identity for real tokens,
    zero for padding -- Eq. 43).

Continue branch (Eq. 44-47, Appendix B)
    Handles the event that the path continues (path length > 1).
    The target path is constructed from (y, W_+) where W_+ ~ m_{h-1}^{theta_bar}
    is a suffix sampled from the FROZEN teacher at the *same* noise level t.
    This is obtained via either:
      (a) Partial reverse integration of the teacher SDE down to time t, OR
      (b) Replay-buffered suffix: cache a clean teacher suffix W_0^+ and corrupt
          it to time t via the forward kernel (Proposition 1 equivalence).

    The target score combines:
      - Analytic score for the current token block (position 0): no gradients needed.
      - STOP-GRADIENT of the teacher score for the suffix blocks (positions 1..h-1).

        U_{h,t}^cont[0]     = -1/sigma_t^2 * (Z_0^cont - alpha_t * phi(y))
        U_{h,t}^cont[1:h-1] = sg[ s_{theta_bar, h-1}(Z_+^cont, t | x') ]

Shapes
------
    B          -- batch size
    h          -- path horizon (number of token blocks)
    token_dim  -- d_tok = 1 + obs_dim + act_dim
    T          -- total diffusion timesteps

References
----------
Algorithm 1 (Training) -- lines 9-22.
Section 5.2 -- Direct Bellmanization of the Reverse Score.
Eq. 41-43   -- stop branch score.
Eq. 44-47   -- continue branch score.
Appendix B  -- suffix sampling equivalence (Proposition 1).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from bpd.models.diffusion import DDPMSchedule


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_coeff(
    a: Tensor,
    t: Tensor,
    broadcast_shape: Tuple[int, ...],
) -> Tensor:
    """Gather 1-D schedule buffer *a* at timesteps *t* and broadcast.

    Args:
        a:               1-D schedule tensor of shape (T+1,) (alphas_bar /
                         sigma convention from DDPMSchedule).
        t:               Integer timestep indices, shape (B,).
        broadcast_shape: Target shape, e.g. (B, h, d).

    Returns:
        Tensor of shape ``broadcast_shape`` with values a[t[b]] for each b.
    """
    out = a.to(t.device)[t]  # (B,)
    while out.dim() < len(broadcast_shape):
        out = out.unsqueeze(-1)
    return out.expand(broadcast_shape)


def _analytic_score(
    z: Tensor,
    w: Tensor,
    t: Tensor,
    sqrt_alphas_bar: Tensor,
    sqrt_one_minus_alphas_bar: Tensor,
) -> Tensor:
    """Analytic score grad_z log q_{t|0}(z | w) = -1/sigma_t^2 * (z - alpha_t * w).

    Used for both the stop branch (all h blocks) and the current-token block
    of the continue branch.

    Args:
        z:                          Noisy tokens, shape (B, h, d) or (B, d).
        w:                          Clean reference tokens, same shape as z.
        t:                          Timestep indices, shape (B,).
        sqrt_alphas_bar:            alpha_t = sqrt(alphabar_t), shape (T+1,).
        sqrt_one_minus_alphas_bar:  sigma_t = sqrt(1 - alphabar_t), shape (T+1,).

    Returns:
        Score tensor of the same shape as z.
    """
    shape = z.shape
    alpha_t = _extract_coeff(sqrt_alphas_bar, t, shape)
    sigma_t = _extract_coeff(sqrt_one_minus_alphas_bar, t, shape)
    sigma_t_sq = sigma_t.pow(2).clamp(min=1e-8)
    return -(z - alpha_t * w) / sigma_t_sq


def _q_sample(
    w: Tensor,
    t: Tensor,
    sqrt_alphas_bar: Tensor,
    sqrt_one_minus_alphas_bar: Tensor,
    noise: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Sample z_t ~ q_{t|0}(. | w) = N(alpha_t * w, sigma_t^2 I).

    Args:
        w:                          Clean path tokens.
        t:                          Timestep indices, shape (B,).
        sqrt_alphas_bar:            Schedule buffer, shape (T+1,).
        sqrt_one_minus_alphas_bar:  Schedule buffer, shape (T+1,).
        noise:                      Optional pre-sampled noise, same shape as w.

    Returns:
        Tuple (z_t, noise) where z_t is the noisy sample.
    """
    if noise is None:
        noise = torch.randn_like(w)
    shape = w.shape
    alpha_t = _extract_coeff(sqrt_alphas_bar, t, shape)
    sigma_t = _extract_coeff(sqrt_one_minus_alphas_bar, t, shape)
    z_t = alpha_t * w + sigma_t * noise
    return z_t, noise


# ---------------------------------------------------------------------------
# Stop Branch
# ---------------------------------------------------------------------------


class StopBranch(nn.Module):
    """Stop branch of the BPD Bellman recursion (Section 5.2, Eq. 41-43).

    Computes the *stop* training target for the score network.

    The stop branch is active when the sampled path terminates at the current
    depth h, which happens with probability (1 - gamma) (geometric law, Eq. 14).
    In this case the target (clean) path is:

        stop_h(y) = (y, pad, ..., pad)    [Eq. 4]

    where pad is represented as the zero vector (phi(bottom) = 0).
    A noisy version Z^stop ~ q_{t|0}^{(h)}(. | stop_h(y)) is constructed by
    applying the blockwise forward kernel to each token independently.

    The analytic score target is (Eq. 43):

        U_{h,t}^stop[j=0]   = -1/sigma_t^2 * (Z^stop_0 - alpha_t * phi(y))
        U_{h,t}^stop[j>0]   = -1/sigma_t^2 * (Z^stop_j - 0)          [padding: phi(pad)=0]

    which simplifies the latter to -z_j / sigma_t^2 for j >= 1.

    Args:
        schedule:  Pre-built DDPMSchedule with noise coefficients.
        token_dim: Token dimensionality d_tok.
    """

    def __init__(self, schedule: DDPMSchedule, token_dim: int) -> None:
        super().__init__()
        self.token_dim = token_dim

        # Register schedule tensors as buffers so .to(device) propagates.
        self.register_buffer("sqrt_alphas_bar", schedule.sqrt_alphas_bar)
        self.register_buffer(
            "sqrt_one_minus_alphas_bar", schedule.sqrt_one_minus_alphas_bar
        )

    def sample_and_target(
        self,
        y: Tensor,
        t: Tensor,
        h: int,
        pad_embed: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the stop-branch noisy sample and analytic score target.

        Constructs the stop path stop_h(y) = (y, 0, ..., 0), applies the
        blockwise forward kernel to obtain Z^stop, then evaluates the
        analytic score at Z^stop (Eq. 43).

        Args:
            y:         Current token embedding, shape (B, token_dim).
                       This is phi(y) -- the embedding of the sampled transition.
            t:         Diffusion timestep indices, shape (B,), in {1, ..., T}.
            h:         Path horizon (number of token blocks per path).  Must be >= 1.
            pad_embed: Embedding of the padding token phi(bottom).  If None,
                       defaults to zeros of shape (token_dim,) (Eq. 43:
                       "phi(bottom) = 0").  The caller may override this for
                       non-zero padding conventions.

        Returns:
            Tuple (z_stop, score_target):

            * z_stop:       Noisy stop path, shape (B, h, token_dim).
                            - z_stop[:, 0, :] ~ N(alpha_t * phi(y), sigma_t^2 I)
                            - z_stop[:, j, :] ~ N(0, sigma_t^2 I) for j >= 1
                              (padding tokens have clean value 0).
            * score_target: Analytic score, shape (B, h, token_dim).
                            - score_target[:, 0, :] = -1/sigma_t^2 * (z0 - alpha_t*phi(y))
                            - score_target[:, j, :] = -z_j / sigma_t^2  for j >= 1

        Raises:
            ValueError: If tensor shapes are inconsistent or h < 1.
        """
        if y.dim() != 2:
            raise ValueError(
                f"y must be 2-D (B, token_dim); got {tuple(y.shape)}"
            )
        B, d = y.shape
        if d != self.token_dim:
            raise ValueError(
                f"y.shape[-1]={d} does not match token_dim={self.token_dim}"
            )
        if t.shape != (B,):
            raise ValueError(
                f"t must have shape ({B},); got {tuple(t.shape)}"
            )
        if h < 1:
            raise ValueError(f"h must be >= 1; got {h}")

        device = y.device

        # ------------------------------------------------------------------
        # 1. Build clean stop path w_{stop} = (phi(y), 0, ..., 0)  [Eq. 4]
        # ------------------------------------------------------------------
        # Determine the padding embedding phi(bottom).
        if pad_embed is None:
            phi_pad = torch.zeros(d, dtype=y.dtype, device=device)
        else:
            phi_pad = pad_embed.to(dtype=y.dtype, device=device).reshape(d)

        # w_{stop} shape: (B, h, d)
        # Position 0: phi(y);  positions 1..h-1: phi(bottom) = phi_pad
        w_stop = phi_pad.unsqueeze(0).unsqueeze(0).expand(B, h, d).clone()
        w_stop = w_stop.contiguous()
        w_stop[:, 0, :] = y  # phi(y) at position 0

        # ------------------------------------------------------------------
        # 2. Sample Z^stop ~ q_{t|0}^{(h)}(. | w_{stop})    [Eq. 41]
        #    Each block is corrupted independently:
        #        z_j = alpha_t * w_{stop,j} + sigma_t * eps_j
        # ------------------------------------------------------------------
        z_stop, _ = _q_sample(
            w_stop,
            t,
            self.sqrt_alphas_bar,
            self.sqrt_one_minus_alphas_bar,
        )

        # ------------------------------------------------------------------
        # 3. Analytic score target U_{h,t}^stop                 [Eq. 43]
        #    = grad_z log q_{t|0}^{(h)}(z_stop | w_{stop})
        #    = -1/sigma_t^2 * (z_stop - alpha_t * w_{stop})
        #
        #    For j=0:   w_{stop,0} = phi(y)   -> analytic residual
        #    For j>=1:  w_{stop,j} = phi(pad) = 0 -> -z_j / sigma_t^2
        # ------------------------------------------------------------------
        score_target = _analytic_score(
            z_stop,
            w_stop,
            t,
            self.sqrt_alphas_bar,
            self.sqrt_one_minus_alphas_bar,
        )

        return z_stop, score_target


# ---------------------------------------------------------------------------
# Continue Branch
# ---------------------------------------------------------------------------


class ContinueBranch(nn.Module):
    """Continue branch of the BPD Bellman recursion (Section 5.2, Eq. 44-47).

    Computes the *continuation* training target for the score network.

    The continue branch is active when the sampled path length exceeds one,
    which happens with probability gamma (geometric law, Eq. 14).
    The target score decomposes as (Eq. 46-47):

        U_{h,t}^cont = [U_0 ; U_+]

    where:
        U_0 = -1/sigma_t^2 * (Z_0^cont - alpha_t * phi(y))    [analytic, Eq. 47]
        U_+ = sg[ s_{theta_bar, h-1}(Z_+^cont, t | x') ]      [teacher, stop-gradient]

    and Z_+^cont is a suffix sample at noise level t from the frozen teacher
    m_{h-1,t}^{theta_bar}(. | x').

    Suffix sampling strategy (Proposition 1 / Appendix B)
    -------------------------------------------------------
    Two equivalent approaches for obtaining Z_+^cont ~ m_{h-1,t}^{theta_bar}(. | x'):

    (a) Partial reverse integration: run the teacher reverse SDE from pure noise
        down to noise level t, conditioning on x'.  This is exact but expensive.

    (b) Replay-buffered corruption (Proposition 1): retrieve a clean suffix
        W_0^+ from the SuffixReplayBuffer for the given x', then corrupt it to
        noise level t via the forward kernel:
            Z_+^cont = alpha_t * W_0^+ + sigma_t * eps,  eps ~ N(0, I)
        This is exact in population (same marginal distribution) and cheap.

    When a replay buffer is provided and a cached clean suffix is available for
    x', strategy (b) is used.  Otherwise strategy (a) is used: the caller must
    supply `teacher_score_fn` that can generate a noisy suffix via the teacher's
    reverse SDE.  The branch calls
        z_plus = teacher_score_fn(z_pure_noise, t, x_prime, mode='reverse')
    if the replay buffer has no entry.  For simplicity and to match Proposition 1,
    the default path when no buffer hit occurs is to sample fresh Gaussian noise
    and corrupt a zero-initialised placeholder -- callers should supply either
    a buffer or a teacher partial-reverse callable.

    Args:
        schedule:  Pre-built DDPMSchedule with noise coefficients.
        token_dim: Token dimensionality d_tok.
    """

    def __init__(self, schedule: DDPMSchedule, token_dim: int) -> None:
        super().__init__()
        self.token_dim = token_dim

        self.register_buffer("sqrt_alphas_bar", schedule.sqrt_alphas_bar)
        self.register_buffer(
            "sqrt_one_minus_alphas_bar", schedule.sqrt_one_minus_alphas_bar
        )

    def _get_suffix_noisy(
        self,
        x_prime: Tensor,
        t: Tensor,
        h: int,
        teacher_score_fn: Callable,
        replay_buffer,
    ) -> Tensor:
        """Obtain Z_+^cont ~ m_{h-1,t}^{theta_bar}(. | x') for each item in batch.

        Implements the two-path suffix sampling strategy of Proposition 1:

        * If a replay buffer is provided and contains a matching entry for x'[b],
          corrupt the cached clean suffix W_0^+ to noise level t via the forward
          kernel (Proposition 1 / Eq. 50).

        * Otherwise, call teacher_score_fn to obtain a noisy suffix directly.
          The callable signature expected is:
              teacher_score_fn(x_prime, t) -> z_plus of shape (B, h-1, token_dim)
          This allows arbitrary teacher sampling strategies (partial reverse SDE,
          distillation, etc.) to be injected by the caller.

        Args:
            x_prime:          Successor conditioning, shape (B, cond_dim).
            t:                Timestep indices, shape (B,).
            h:                Current path horizon (suffix has length h-1).
            teacher_score_fn: Callable for obtaining noisy suffix from teacher.
            replay_buffer:    SuffixReplayBuffer instance or None.

        Returns:
            Noisy suffix tensor Z_+^cont of shape (B, h-1, token_dim).
        """
        B = x_prime.shape[0]
        h_suffix = h - 1
        device = x_prime.device

        if h_suffix == 0:
            # Degenerate case: h=1 has no suffix.  Continue branch should not
            # be called with h=1, but we return an empty tensor gracefully.
            return torch.zeros(B, 0, self.token_dim, dtype=x_prime.dtype, device=device)

        if replay_buffer is not None:
            # ------------------------------------------------------------------
            # Strategy (b): Replay-buffered corruption (Proposition 1 / Eq. 50)
            # For each batch item, try to retrieve a cached clean suffix.
            # Fall back to teacher_score_fn for cache misses.
            # ------------------------------------------------------------------
            z_plus_list = []
            # We handle the batch item by item to support partial cache hits.
            for b in range(B):
                x_b = x_prime[b]  # (cond_dim,)
                t_b = t[b : b + 1]  # (1,)

                cached = replay_buffer.sample_or_none(x_b)
                if cached is not None:
                    # Corrupt the clean suffix to noise level t via forward kernel.
                    # cached has shape (h_suffix, token_dim) on CPU.
                    w_plus_clean = cached.to(dtype=torch.float32, device=device)
                    if w_plus_clean.shape[0] == h_suffix:
                        # Standard case: suffix length matches.
                        w_plus_clean = w_plus_clean.unsqueeze(0)  # (1, h_suffix, d)
                    else:
                        # Suffix length mismatch (e.g., buffer was built for a
                        # different horizon).  Truncate or pad to h_suffix.
                        if w_plus_clean.shape[0] > h_suffix:
                            w_plus_clean = w_plus_clean[:h_suffix].unsqueeze(0)
                        else:
                            # Pad with zeros.
                            pad = torch.zeros(
                                h_suffix - w_plus_clean.shape[0],
                                self.token_dim,
                                dtype=w_plus_clean.dtype,
                                device=device,
                            )
                            w_plus_clean = torch.cat([w_plus_clean, pad], dim=0).unsqueeze(0)

                    z_plus_b, _ = _q_sample(
                        w_plus_clean,
                        t_b,
                        self.sqrt_alphas_bar,
                        self.sqrt_one_minus_alphas_bar,
                    )  # (1, h_suffix, d)
                    z_plus_list.append(z_plus_b)
                else:
                    # Cache miss: fall back to teacher_score_fn.
                    x_b_batch = x_b.unsqueeze(0)  # (1, cond_dim)
                    z_plus_b = teacher_score_fn(x_b_batch, t_b)  # (1, h_suffix, d)
                    if z_plus_b.shape[1] != h_suffix:
                        raise ValueError(
                            f"teacher_score_fn returned suffix of length "
                            f"{z_plus_b.shape[1]}, expected h-1={h_suffix}"
                        )
                    z_plus_list.append(z_plus_b)

            z_plus = torch.cat(z_plus_list, dim=0)  # (B, h_suffix, d)
        else:
            # ------------------------------------------------------------------
            # Strategy (a) / default: Use teacher_score_fn for the full batch.
            # Expected signature: teacher_score_fn(x_prime, t) -> (B, h-1, d)
            # ------------------------------------------------------------------
            z_plus = teacher_score_fn(x_prime, t)  # (B, h_suffix, d)
            if z_plus.shape != (B, h_suffix, self.token_dim):
                raise ValueError(
                    f"teacher_score_fn must return shape "
                    f"({B}, {h_suffix}, {self.token_dim}); got {tuple(z_plus.shape)}"
                )

        return z_plus

    def sample_and_target(
        self,
        y: Tensor,
        x_prime: Tensor,
        t: Tensor,
        h: int,
        teacher_score_fn: Callable,
        replay_buffer=None,
        pad_embed: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the continue-branch noisy sample and score target.

        Implements Algorithm 1 lines 15-22 (continue branch):

        1. Corrupt current token y to noise level t:
               Z_0^cont = alpha_t * phi(y) + sigma_t * eps_0       [Eq. 44]
        2. Obtain noisy suffix Z_+^cont ~ m_{h-1,t}^{theta_bar}(. | x')
           via replay (Proposition 1) or teacher partial reverse SDE.    [Eq. 45]
        3. Assemble the full noisy path Z^cont = (Z_0^cont ; Z_+^cont). [Eq. 44]
        4. Evaluate score target:
               U_0     = -1/sigma_t^2 * (Z_0^cont - alpha_t * phi(y))  [Eq. 47]
               U_+     = sg[ s_{theta_bar,h-1}(Z_+^cont, t | x') ]     [Eq. 46]
               U^cont  = (U_0 ; U_+)

        The STOP-GRADIENT on U_+ is enforced by detaching Z_+^cont before
        passing it to teacher_score_fn for the target evaluation.

        Args:
            y:                Current token embedding phi(y), shape (B, token_dim).
            x_prime:          Next conditioning pair (s', a'), shape (B, cond_dim)
                              where cond_dim = state_dim + act_dim.
            t:                Diffusion timestep indices, shape (B,).
            h:                Current path horizon.  Must be >= 2 (horizon h=1 has
                              no suffix and should use StopBranch exclusively).
            teacher_score_fn: Callable representing the FROZEN teacher score network
                              s_{theta_bar, h-1}.  Two supported calling conventions:

                              (a) Suffix sampling (no buffer hit):
                                      teacher_score_fn(x_prime, t) -> z_plus
                                  Returns a noisy suffix of shape (B, h-1, token_dim)
                                  already at noise level t (via partial reverse SDE).

                              (b) Score evaluation:
                                      teacher_score_fn(z_plus, t, x_prime, mode='score')
                                  Returns the teacher score at z_plus, shape
                                  (B, h-1, token_dim).

                              Convention:
                                  - If replay_buffer is None, teacher_score_fn is
                                    called as (x_prime, t) to obtain z_plus, AND
                                    then called as (z_plus, t, x_prime) to obtain
                                    the teacher score (stop-gradient applied by
                                    detaching z_plus).
                                  - If replay_buffer provides a hit, z_plus is
                                    obtained via corruption, and teacher_score_fn
                                    is called as (z_plus_detached, t, x_prime)
                                    for the teacher score.

                              In either case, the caller is responsible for
                              ensuring teacher_score_fn uses frozen (no-grad)
                              parameters.  We additionally wrap the call in
                              torch.no_grad() to prevent gradient flow.

            replay_buffer:    Optional SuffixReplayBuffer.  If provided and a
                              cached clean suffix for x'[b] is found, strategy (b)
                              from Proposition 1 is used for that item.
            pad_embed:        Embedding phi(bottom) for the padding token.
                              Defaults to zeros (Eq. 43).

        Returns:
            Tuple (z_cont, score_target):

            * z_cont:        Concatenated noisy path (Z_0^cont ; Z_+^cont),
                             shape (B, h, token_dim).
                             - z_cont[:, 0, :]   = Z_0^cont (noisy current token)
                             - z_cont[:, 1:, :]  = Z_+^cont (noisy teacher suffix)

            * score_target:  Combined score target, shape (B, h, token_dim).
                             - score_target[:, 0, :]  = U_0  (analytic, with grad)
                             - score_target[:, 1:, :] = U_+  (teacher, stop-gradient)

        Raises:
            ValueError: If tensor shapes are inconsistent or h < 2.
        """
        if y.dim() != 2:
            raise ValueError(
                f"y must be 2-D (B, token_dim); got {tuple(y.shape)}"
            )
        B, d = y.shape
        if d != self.token_dim:
            raise ValueError(
                f"y.shape[-1]={d} does not match token_dim={self.token_dim}"
            )
        if x_prime.dim() != 2 or x_prime.shape[0] != B:
            raise ValueError(
                f"x_prime must be 2-D with batch size {B}; got {tuple(x_prime.shape)}"
            )
        if t.shape != (B,):
            raise ValueError(
                f"t must have shape ({B},); got {tuple(t.shape)}"
            )
        if h < 2:
            raise ValueError(
                f"ContinueBranch requires h >= 2 (suffix has at least 1 block); "
                f"got h={h}.  Use StopBranch for h=1."
            )

        device = y.device
        h_suffix = h - 1

        # Resolve pad_embed (unused in continue branch but kept for API symmetry).
        if pad_embed is not None:
            pad_embed = pad_embed.to(dtype=y.dtype, device=device)

        # ------------------------------------------------------------------
        # Step 1: Corrupt current token to noise level t  [Eq. 44]
        #         Z_0^cont ~ q_{t|0}(. | phi(y))
        # ------------------------------------------------------------------
        y_expanded = y.unsqueeze(1)  # (B, 1, d)  -- treat as 1-block path
        z_0_cont, _ = _q_sample(
            y_expanded,
            t,
            self.sqrt_alphas_bar,
            self.sqrt_one_minus_alphas_bar,
        )
        # z_0_cont shape: (B, 1, d)

        # ------------------------------------------------------------------
        # Step 2: Obtain noisy suffix Z_+^cont ~ m_{h-1,t}^{theta_bar}(. | x')
        #         via replay buffer (Proposition 1) or teacher_score_fn.
        #         [Eq. 45]
        # ------------------------------------------------------------------
        # Wrap in no_grad to prevent gradients from flowing through the teacher.
        with torch.no_grad():
            z_plus_cont = self._get_suffix_noisy(
                x_prime, t, h, teacher_score_fn, replay_buffer
            )  # (B, h_suffix, d)

        # ------------------------------------------------------------------
        # Step 3: Assemble Z^cont = (Z_0^cont ; Z_+^cont)  [Eq. 44]
        # ------------------------------------------------------------------
        z_cont = torch.cat([z_0_cont, z_plus_cont], dim=1)  # (B, h, d)

        # ------------------------------------------------------------------
        # Step 4a: Analytic score for the current-token block  [Eq. 47]
        #          U_0 = -1/sigma_t^2 * (Z_0^cont - alpha_t * phi(y))
        # ------------------------------------------------------------------
        score_0 = _analytic_score(
            z_0_cont,  # (B, 1, d)
            y_expanded,  # clean reference phi(y), (B, 1, d)
            t,
            self.sqrt_alphas_bar,
            self.sqrt_one_minus_alphas_bar,
        )
        # score_0 shape: (B, 1, d)

        # ------------------------------------------------------------------
        # Step 4b: Teacher score for the suffix blocks  [Eq. 46]
        #          U_+ = sg[ s_{theta_bar, h-1}(Z_+^cont, t | x') ]
        #
        # We detach Z_+^cont and evaluate inside no_grad to implement
        # the stop-gradient (sg[.]) operation.
        # ------------------------------------------------------------------
        z_plus_sg = z_plus_cont.detach()  # stop-gradient

        with torch.no_grad():
            # teacher_score_fn(z_plus, t, x_prime) -> score of shape (B, h_suffix, d)
            score_plus_raw = teacher_score_fn(z_plus_sg, t, x_prime)

        # Wrap the teacher score in a tensor that has no grad_fn (already
        # detached by no_grad, but make explicit for clarity).
        score_plus = score_plus_raw.detach()  # (B, h_suffix, d)

        if score_plus.shape != (B, h_suffix, self.token_dim):
            raise ValueError(
                f"teacher_score_fn (score mode) must return shape "
                f"({B}, {h_suffix}, {self.token_dim}); got {tuple(score_plus.shape)}"
            )

        # ------------------------------------------------------------------
        # Step 4c: Concatenate to form the full target U^cont = (U_0 ; U_+)
        # ------------------------------------------------------------------
        score_target = torch.cat([score_0, score_plus], dim=1)  # (B, h, d)

        return z_cont, score_target
