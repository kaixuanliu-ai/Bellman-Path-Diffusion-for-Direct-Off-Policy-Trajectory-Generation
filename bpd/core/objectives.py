"""
Bellman diffusion objective L_h(theta; theta_bar) for Bellman Path Diffusion.

Implements Eq. 48 from Section 5.3 of:
  "Bellman Path Diffusion for Direct Off-Policy Trajectory Generation"

KEY EQUATION (Eq. 48):
    L_h(theta; theta_bar) = E_{X,t}[
        (1 - gamma) * E[lambda(t) * ||s_{theta,h}(Z^stop, t|X) - U_{h,t}^stop||^2]
        +   gamma   * E[lambda(t) * ||s_{theta,h}(Z^cont, t|X) - U_{h,t}^cont||^2]
    ]

where:
  - C ~ Bernoulli(gamma) selects which branch is evaluated
  - stop branch (C=0): path = stop_h(y),  target = analytic blockwise score of q_{t|0}^{(h)}
  - cont branch (C=1): path = push_h(y, W_+),  target = (1/sigma_t^2) * alpha_t * y_0
                        + gamma * E_teacher[score of m_{h-1,t}^pi at Z^cont_{1:}|X']
  - lambda(t) is an optional loss weight (see get_loss_weight)

REMARK 2 (paper): This is NOT a regulariser – removing either branch changes the
population minimiser and thus the target path measure.

THEOREM 2 (paper): The population minimiser of L_h is exactly nabla_z log m_{h,t}^pi,
given an exact teacher.

Score network convention:
  The score network s_{theta,h}(z, t | x) predicts the SCORE (not the noise).
  Internally, BlockwiseDiffusion.blockwise_score_target already returns the score.
  The teacher_score_fn is also expected to return a score.

Shapes used throughout:
  B          – batch size
  h          – horizon (number of token positions in the path)
  token_dim  – d_tok = 1 + obs_dim + act_dim
  T          – total diffusion steps
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bpd.models.diffusion import DDPMSchedule, BlockwiseDiffusion
from bpd.utils.arrays import DataBatch
from bpd.data.replay import SuffixReplayBuffer
from bpd.core.path import stop_map, push_map


# ---------------------------------------------------------------------------
# Loss weight function
# ---------------------------------------------------------------------------


def get_loss_weight(
    t: Tensor,
    schedule: DDPMSchedule,
    mode: str = "uniform",
    min_snr_gamma: float = 5.0,
) -> Tensor:
    """Compute the per-timestep loss weight lambda(t) for the diffusion objective.

    Three weighting modes are supported:

    ``"uniform"``
        lambda(t) = 1 for all t.  This is the standard DDPM loss (Ho et al. 2020).

    ``"snr"``
        lambda(t) = alpha_t^2 / sigma_t^2 = alpha_bar_t / (1 - alpha_bar_t).
        This is the signal-to-noise ratio weighting, corresponding to a
        variance-weighted denoising score matching objective
        (Song et al. 2021 / Kingma et al. 2021).

    ``"min_snr_gamma"``
        lambda(t) = min(SNR(t), gamma_snr) where gamma_snr = *min_snr_gamma*.
        Introduced by Hang et al. 2023 ("Efficient Diffusion Training via
        Min-SNR Weighting Strategy") to stabilise training at low SNR timesteps.

    Args:
        t:            Integer timestep indices of shape (B,), values in
                      {1, ..., T}.
        schedule:     Pre-built :class:`~bpd.models.diffusion.DDPMSchedule`.
                      Must have ``alphas_bar`` tensor of shape (T+1,).
        mode:         Weighting mode: ``"uniform"``, ``"snr"``, or
                      ``"min_snr_gamma"``.
        min_snr_gamma: Clipping threshold used when ``mode="min_snr_gamma"``.
                       Corresponds to gamma_snr in Hang et al. 2023 (default 5).

    Returns:
        Loss weight tensor of shape (B,) in float32.

    Raises:
        ValueError: If *mode* is not one of the three supported values.
    """
    if mode == "uniform":
        return torch.ones(t.shape[0], dtype=torch.float32, device=t.device)

    # Gather alpha_bar_t for each sample in the batch.  alphas_bar has shape
    # (T+1,) with index 0 = t=0 (no noise), index t = alpha_bar at step t.
    alpha_bar_t = schedule.alphas_bar.to(t.device)[t]  # (B,)
    # Clamp to avoid division by zero at t=0 (alpha_bar_0 = 1 => sigma = 0).
    one_minus_ab = (1.0 - alpha_bar_t).clamp(min=1e-8)
    snr = alpha_bar_t / one_minus_ab  # alpha_t^2 / sigma_t^2

    if mode == "snr":
        return snr.float()

    if mode == "min_snr_gamma":
        return snr.clamp(max=min_snr_gamma).float()

    raise ValueError(
        f"Unknown loss weight mode '{mode}'. "
        "Choose from 'uniform', 'snr', or 'min_snr_gamma'."
    )


# ---------------------------------------------------------------------------
# Bellman Diffusion Loss
# ---------------------------------------------------------------------------


class BellmanDiffusionLoss(nn.Module):
    """Bellman diffusion objective L_h(theta; theta_bar) from Eq. 48 of the paper.

    At each call the loss randomly selects a branch:

    * **Stop branch** (probability 1 - gamma):  The path is ``stop_h(y)``.
      The score target is the analytic blockwise score of the forward kernel
      ``q_{t|0}^{(h)}(z | stop_h(y))``, which decomposes into a sum over
      independent Gaussian scores (Eq. 43).

    * **Continue branch** (probability gamma):  The path is ``push_h(y, W_+)``
      where ``W_+`` is a suffix of length h-1 drawn from one of two sources:

        1. A clean teacher-rollout from the replay buffer (if available for this
           conditioning context ``x'``).
        2. A fresh teacher sample obtained by calling ``teacher_score_fn`` as a
           full reverse-diffusion chain on the sub-horizon h-1 problem.

      The score target for the continue branch is:

          U_{h,t}^cont(z, x) = (1/sigma_t^2) * alpha_t * y * [mask_pos0]
                              + gamma * E_{Z' ~ q_{t|0}(.|W_+)}[
                                    teacher_score_fn(Z'[1:], t, x') ]
                              concatenated along the h dimension.

      In practice, since W_+ is fixed (not a random variable once sampled), the
      expectation over ``q_{t|0}(.|W_+)`` at position 0 is just the Gaussian
      score of that position's kernel.

      Concretely, the target for the full noisy path ``Z = [Z_0, Z_1, ..., Z_{h-1}]``
      under ``push_h(y, W_+)`` is:

          U^cont[0]   = analytic score at position 0  (Gaussian: -(Z_0 - alpha_t*y) / sigma_t^2)
          U^cont[1:h] = gamma * teacher_score_fn(Z[1:h], t, x')

      This implements the Bellman equation for the score (Eq. 44-48 of the paper):

          nabla_z log m_{h,t}^pi(z, x)
            = (1 - gamma) * nabla_z log q_{t|0}^{(h)}(z | stop_h(y))
            + gamma       * nabla_z log m_{h,t}^pi(z | push_h(y, W_+))

      where the second term factors into position 0 (analytic) plus positions 1:h
      (teacher recursion, weighted by gamma).

    Args:
        gamma:           Geometric discount / continuation probability in (0, 1).
        schedule:        Pre-built :class:`~bpd.models.diffusion.DDPMSchedule`.
        token_dim:       d_tok = 1 + obs_dim + act_dim.
        loss_weight_fn:  Optional callable ``(t, schedule) -> Tensor`` returning
                         per-sample weights lambda(t) of shape (B,).  If ``None``
                         defaults to uniform weighting (lambda = 1).

    Example::

        schedule = DDPMSchedule.make_cosine(T=1000)
        loss_fn = BellmanDiffusionLoss(gamma=0.99, schedule=schedule, token_dim=d_tok)
        loss = loss_fn(score_net, teacher_score_fn, batch, h=4)
        loss.backward()
    """

    def __init__(
        self,
        gamma: float,
        schedule: DDPMSchedule,
        token_dim: int,
        loss_weight_fn: Optional[Callable[[Tensor, DDPMSchedule], Tensor]] = None,
    ) -> None:
        super().__init__()

        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}")
        if token_dim < 1:
            raise ValueError(f"token_dim must be >= 1; got {token_dim}")

        self.gamma = gamma
        self.schedule = schedule
        self.token_dim = token_dim
        self.loss_weight_fn = loss_weight_fn

        # Construct the blockwise diffusion helper for q_sample and score target.
        self._diffusion = BlockwiseDiffusion(schedule, token_dim)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        score_net: nn.Module,
        teacher_score_fn: Callable[[Tensor, Tensor, Tensor, int], Tensor],
        batch: DataBatch,
        h: int,
        replay_buffer: Optional[SuffixReplayBuffer] = None,
    ) -> Tensor:
        """Compute the scalar Bellman diffusion loss for a batch of transitions.

        For each sample in the batch this method:
        1. Samples a branch indicator C ~ Bernoulli(gamma).
        2. Constructs the appropriate path and score target for each branch.
        3. Evaluates the student score network on the noisy path.
        4. Returns the weighted mean-squared error.

        Args:
            score_net:
                Student score network (the model being trained).  Expected
                signature: ``score_net(z_t, t, x, h) -> Tensor`` where

                * ``z_t``  has shape ``(B, h, token_dim)`` (noisy path),
                * ``t``    has shape ``(B,)``              (timestep indices, long),
                * ``x``    has shape ``(B, obs_dim + act_dim)``
                           (conditioning context – current state-action pair), and
                * ``h``    is a Python int (horizon).

                Returns a score tensor of shape ``(B, h, token_dim)``.

            teacher_score_fn:
                Teacher / target score network (EMA copy or fixed pre-trained
                model, denoted ``theta_bar`` in the paper).  Called with the
                same signature as ``score_net`` but treated as a *fixed* function
                (no gradient flow through teacher parameters):
                ``teacher_score_fn(z_t, t, x, h_sub) -> Tensor``.

                For the continue branch it is called on paths of length h-1
                with context ``x' = (s', a')``.

            batch:
                :class:`~bpd.utils.arrays.DataBatch` namedtuple with fields:

                * ``state``      – ``(B, obs_dim)``  current observation  s
                * ``action``     – ``(B, act_dim)``  current action        a
                * ``reward``     – ``(B,)``           reward                r
                * ``next_state`` – ``(B, obs_dim)``  next observation      s'
                * ``done``       – ``(B,)``           terminal flag

                The token ``y = (r, s', a')`` is constructed internally using
                ``next_state`` and a re-use of ``action`` as the *next* action
                (see Note below).

            h:
                Path horizon h >= 1.  The path tensors constructed here have
                shape ``(B, h, token_dim)``.

            replay_buffer:
                Optional :class:`~bpd.data.replay.SuffixReplayBuffer`.  When
                provided, the continue branch attempts to look up a cached clean
                suffix for each sample before falling back to a teacher rollout.
                When ``None``, fresh teacher samples are always used.

        Returns:
            Scalar loss tensor (mean over batch and token dimensions).

        Note on token construction:
            The paper defines the token as ``y = (r, s', a')`` where ``a' ~
            pi_e(.|s')``.  This implementation uses ``batch.action`` as a proxy
            for ``a'`` (the offline dataset action at the *next* state) to avoid
            requiring an explicit evaluation-policy query inside the loss.  In
            practice the dataset action at ``s'`` is stored as the ``action``
            field of the *following* transition.  If you need strict
            ``a' ~ pi_e(.|s')`` you should pre-compute the token in the
            DataLoader and pass it directly.
        """
        device = batch.state.device
        B = batch.state.shape[0]

        # ------------------------------------------------------------------ #
        # 1.  Build the current token y = (r, s', a')                        #
        #     Layout: [r | next_state | action]                               #
        # ------------------------------------------------------------------ #
        reward = batch.reward.reshape(B, 1).to(dtype=torch.float32, device=device)
        next_state = batch.next_state.to(dtype=torch.float32, device=device)
        action = batch.action.to(dtype=torch.float32, device=device)

        # y : (B, token_dim)
        y = torch.cat([reward, next_state, action], dim=-1)

        # Conditioning context for the current token: x = (s, a) -> (B, obs+act)
        x = torch.cat([batch.state.to(device), action], dim=-1)  # (B, obs+act)

        # Conditioning context for the suffix: x' = (s', a') -> (B, obs+act)
        x_prime = torch.cat([next_state, action], dim=-1)  # (B, obs+act)

        # ------------------------------------------------------------------ #
        # 2.  Sample a single shared timestep t ~ Uniform{1,...,T}            #
        # ------------------------------------------------------------------ #
        t = torch.randint(
            1, self.schedule.T + 1, (B,), dtype=torch.long, device=device
        )  # (B,)

        # ------------------------------------------------------------------ #
        # 3.  Sample branch indicator C ~ Bernoulli(gamma)                   #
        #     C=0 → stop branch (prob 1-gamma)                               #
        #     C=1 → cont branch (prob gamma)                                 #
        # ------------------------------------------------------------------ #
        C = torch.bernoulli(
            torch.full((B,), self.gamma, device=device)
        ).bool()  # (B,)

        # ------------------------------------------------------------------ #
        # 4.  Compute per-sample loss weights lambda(t)                      #
        # ------------------------------------------------------------------ #
        if self.loss_weight_fn is not None:
            lam = self.loss_weight_fn(t, self.schedule)  # (B,)
        else:
            lam = torch.ones(B, dtype=torch.float32, device=device)
        lam = lam.to(device=device, dtype=torch.float32)

        # ------------------------------------------------------------------ #
        # 5.  Compute branch losses                                           #
        # ------------------------------------------------------------------ #
        # Initialise loss accumulator
        loss = torch.zeros(B, dtype=torch.float32, device=device)

        # -- Stop branch (C == False) ----------------------------------------
        stop_mask = ~C  # (B,)
        if stop_mask.any():
            stop_loss = self._stop_loss(
                score_net=score_net,
                y=y[stop_mask],
                t=t[stop_mask],
                h=h,
                x=x[stop_mask],
            )  # (B_stop,)
            loss[stop_mask] = stop_loss

        # -- Continue branch (C == True) ------------------------------------
        cont_mask = C  # (B,)
        if h > 1 and cont_mask.any():
            cont_loss = self._cont_loss(
                score_net=score_net,
                teacher_score_fn=teacher_score_fn,
                y=y[cont_mask],
                x_prime=x_prime[cont_mask],
                t=t[cont_mask],
                h=h,
                x=x[cont_mask],
                replay_buffer=replay_buffer,
            )  # (B_cont,)
            loss[cont_mask] = cont_loss
        elif h == 1 and cont_mask.any():
            # At h=1 the push map is undefined; the continue branch collapses
            # to the stop branch (single-step horizon can only stop).
            stop_loss_fallback = self._stop_loss(
                score_net=score_net,
                y=y[cont_mask],
                t=t[cont_mask],
                h=h,
                x=x[cont_mask],
            )
            loss[cont_mask] = stop_loss_fallback

        # ------------------------------------------------------------------ #
        # 6.  Apply loss weights and return scalar                            #
        # ------------------------------------------------------------------ #
        # loss : (B,),  lam : (B,)
        weighted = lam * loss  # (B,)
        return weighted.mean()

    # ------------------------------------------------------------------
    # Stop branch
    # ------------------------------------------------------------------

    def _stop_loss(
        self,
        score_net: nn.Module,
        y: Tensor,
        t: Tensor,
        h: int,
        x: Tensor,
    ) -> Tensor:
        """Compute the stop-branch MSE for a sub-batch.

        The stop path is ``w = stop_h(y)`` which places ``y`` at position 0
        and fills positions 1:h with zero padding tokens.

        The score target is the analytic blockwise Gaussian score:

            U_{h,t}^stop(z, x) = nabla_z log q_{t|0}^{(h)}(z | stop_h(y))
                                = -(z - alpha_t * stop_h(y)) / sigma_t^2

        (Eq. 43, combined with Eq. 24 giving block-independence.)

        Args:
            score_net: Student network, callable as ``score_net(z_t, t, x, h)``.
            y:         Token tensor of shape ``(B_s, token_dim)``.
            t:         Timestep indices of shape ``(B_s,)``.
            h:         Path horizon.
            x:         Conditioning context of shape ``(B_s, obs+act)``.

        Returns:
            Per-sample MSE of shape ``(B_s,)`` – NOT yet multiplied by lambda.
        """
        B_s = y.shape[0]
        device = y.device

        # Construct the stop path w = stop_h(y): (B_s, h, token_dim)
        # Position 0 = y, positions 1:h = zeros.
        w_stop = torch.zeros(B_s, h, self.token_dim, dtype=torch.float32, device=device)
        w_stop[:, 0, :] = y  # broadcast y into all samples

        # Apply the blockwise forward kernel: z_t ~ q_{t|0}^{(h)}(. | w_stop)
        # noise is sampled internally
        z_t = self._diffusion.q_sample_blockwise(w_stop, t)  # (B_s, h, token_dim)

        # Analytic score target U_{h,t}^stop(z_t, x) shape (B_s, h, token_dim)
        score_target = self._diffusion.blockwise_score_target(z_t, w_stop, t)

        # Student score prediction (gradient enabled for theta)
        score_pred = score_net(z_t, t, x, h)  # (B_s, h, token_dim)

        # MSE per sample: mean over (h, token_dim) dimensions
        mse = ((score_pred - score_target) ** 2).mean(dim=(1, 2))  # (B_s,)
        return mse

    # ------------------------------------------------------------------
    # Continue branch
    # ------------------------------------------------------------------

    def _cont_loss(
        self,
        score_net: nn.Module,
        teacher_score_fn: Callable[[Tensor, Tensor, Tensor, int], Tensor],
        y: Tensor,
        x_prime: Tensor,
        t: Tensor,
        h: int,
        x: Tensor,
        replay_buffer: Optional[SuffixReplayBuffer],
    ) -> Tensor:
        """Compute the continue-branch MSE for a sub-batch.

        The continue path is ``w = push_h(y, W_+)`` where ``W_+`` is a clean
        suffix of length ``h-1`` distributed as ``M_{h-1}^pi(. | x')``.

        The score target is:

            U_{h,t}^cont(z, x) [position 0]
                = -(z[:, 0, :] - alpha_t * y) / sigma_t^2
                  (analytic Gaussian score for the head token)

            U_{h,t}^cont(z, x) [positions 1:h]
                = gamma * teacher_score_fn(z[:, 1:, :], t, x', h-1)
                  (teacher score on the sub-horizon suffix, scaled by gamma)

        This implements the recursive Bellman structure (Eq. 48):
            nabla_{z_{1:}} log m_{h,t}^pi(z|x)
                = gamma * nabla_{z_{1:}} log m_{h-1,t}^pi(z_{1:}|x')

        Steps:
        1.  For each sample, attempt to retrieve a cached clean suffix W_+
            from *replay_buffer*; otherwise generate one from the teacher SDE.
        2.  Construct the noisy pushed path by applying ``q_sample_blockwise``
            to the full path ``push_h(y, W_+)``.
        3.  Compute the analytic score at position 0 (head token).
        4.  Evaluate the teacher on the suffix ``z[:, 1:, :]`` to get the
            gamma-weighted score target for positions 1:h.
        5.  Concatenate and compute MSE against the student prediction.

        Args:
            score_net:        Student network ``(z_t, t, x, h) -> Tensor``.
            teacher_score_fn: Teacher network ``(z_t, t, x, h_sub) -> Tensor``.
            y:                Head token ``(B_c, token_dim)``.
            x_prime:          Suffix conditioning ``(s', a')``, shape
                              ``(B_c, obs+act)``.
            t:                Timestep indices ``(B_c,)``.
            h:                Full path horizon (suffix has length h-1).
            x:                Full-path conditioning ``(s, a)``, shape
                              ``(B_c, obs+act)``.
            replay_buffer:    Optional replay buffer for cached suffixes.

        Returns:
            Per-sample MSE of shape ``(B_c,)`` – NOT yet multiplied by lambda.
        """
        B_c = y.shape[0]
        device = y.device
        h_sub = h - 1  # suffix horizon

        # ------------------------------------------------------------------ #
        # 1.  Gather or generate clean suffixes W_+ of shape (B_c, h-1, d)   #
        # ------------------------------------------------------------------ #
        W_plus = self._get_clean_suffix(
            teacher_score_fn=teacher_score_fn,
            x_prime=x_prime,
            h_sub=h_sub,
            device=device,
            replay_buffer=replay_buffer,
        )  # (B_c, h_sub, token_dim)

        # ------------------------------------------------------------------ #
        # 2.  Build the full pushed path w = push_h(y, W_+)                  #
        #     w[:, 0, :] = y                                                  #
        #     w[:, 1:h, :] = W_+                                              #
        # ------------------------------------------------------------------ #
        w_cont = torch.zeros(B_c, h, self.token_dim, dtype=torch.float32, device=device)
        w_cont[:, 0, :] = y
        w_cont[:, 1:, :] = W_plus.to(device=device)  # (B_c, h-1, token_dim)

        # ------------------------------------------------------------------ #
        # 3.  Apply blockwise forward kernel to the full path                 #
        #     z_t ~ q_{t|0}^{(h)}(. | w_cont)                                #
        # ------------------------------------------------------------------ #
        z_t = self._diffusion.q_sample_blockwise(w_cont, t)  # (B_c, h, token_dim)

        # ------------------------------------------------------------------ #
        # 4.  Compute the score target for position 0 (head token)           #
        #     = analytic score of q_{t|0}(z_0 | y)                           #
        # ------------------------------------------------------------------ #
        # Extract schedule coefficients alpha_t and sigma_t for each sample.
        alpha_t = self._diffusion.sqrt_alphas_bar[t]  # (B_c,)  alpha_t = sqrt(abar_t)
        sigma_t = self._diffusion.sqrt_one_minus_alphas_bar[t]  # (B_c,)

        # Reshape for broadcast: (B_c, 1, token_dim) per sample
        alpha_t_bc = alpha_t.view(B_c, 1, 1)  # (B_c, 1, 1)
        sigma_t_sq_bc = (sigma_t ** 2).view(B_c, 1, 1).clamp(min=1e-8)

        # Analytic Gaussian score at position 0:
        #   -(z_t[:, 0, :] - alpha_t * y) / sigma_t^2
        z0 = z_t[:, 0:1, :]   # (B_c, 1, token_dim)
        y_bc = y.unsqueeze(1)  # (B_c, 1, token_dim)
        head_score_target = -(z0 - alpha_t_bc * y_bc) / sigma_t_sq_bc  # (B_c, 1, d)

        # ------------------------------------------------------------------ #
        # 5.  Compute teacher score for suffix positions 1:h                  #
        #     target[1:] = gamma * teacher_score(z_t[:, 1:, :], t, x', h-1) #
        # ------------------------------------------------------------------ #
        z_suffix = z_t[:, 1:, :]  # (B_c, h-1, token_dim)

        with torch.no_grad():
            teacher_suffix_score = teacher_score_fn(
                z_suffix, t, x_prime, h_sub
            )  # (B_c, h_sub, token_dim)

        suffix_score_target = self.gamma * teacher_suffix_score  # (B_c, h-1, token_dim)

        # ------------------------------------------------------------------ #
        # 6.  Concatenate full target U_{h,t}^cont along the h dimension     #
        # ------------------------------------------------------------------ #
        score_target = torch.cat(
            [head_score_target, suffix_score_target], dim=1
        )  # (B_c, h, token_dim)

        # ------------------------------------------------------------------ #
        # 7.  Student score prediction on the full noisy path                 #
        # ------------------------------------------------------------------ #
        score_pred = score_net(z_t, t, x, h)  # (B_c, h, token_dim)

        # ------------------------------------------------------------------ #
        # 8.  MSE per sample (mean over h and token_dim)                      #
        # ------------------------------------------------------------------ #
        mse = ((score_pred - score_target) ** 2).mean(dim=(1, 2))  # (B_c,)
        return mse

    # ------------------------------------------------------------------
    # Suffix generation / retrieval helper
    # ------------------------------------------------------------------

    def _get_clean_suffix(
        self,
        teacher_score_fn: Callable[[Tensor, Tensor, Tensor, int], Tensor],
        x_prime: Tensor,
        h_sub: int,
        device: torch.device,
        replay_buffer: Optional[SuffixReplayBuffer],
    ) -> Tensor:
        """Gather clean suffix samples W_+ for the continue branch.

        Attempts to retrieve each suffix from the replay buffer first.  For
        samples without a cached entry, calls the teacher to produce a clean
        suffix via full reverse-diffusion from ``t=T`` to ``t=1``.

        The teacher reverse chain is:
            W_+ ~ M_{h-1}^pi(. | x') approx int p_teacher(z_0 | x') dz_T ... dz_1

        Args:
            teacher_score_fn: Teacher callable ``(z_t, t, x, h_sub) -> score``.
            x_prime:          Suffix conditioning ``(B_c, obs+act)``.
            h_sub:            Suffix horizon h-1.
            device:           Target compute device.
            replay_buffer:    Optional buffer for caching clean suffixes.

        Returns:
            Clean suffix tensor of shape ``(B_c, h_sub, token_dim)``.
        """
        B_c = x_prime.shape[0]

        if h_sub <= 0:
            # Edge case: h=1 means no suffix needed; return empty.
            return torch.zeros(B_c, 0, self.token_dim, dtype=torch.float32, device=device)

        W_plus = torch.zeros(B_c, h_sub, self.token_dim, dtype=torch.float32, device=device)

        # Track which samples still need a fresh teacher rollout.
        needs_rollout_mask = torch.ones(B_c, dtype=torch.bool, device=device)

        # Attempt replay buffer lookup for each sample.
        if replay_buffer is not None:
            for i in range(B_c):
                cached = replay_buffer.sample_or_none(x_prime[i])
                if cached is not None:
                    # cached shape: (h_sub, token_dim) on CPU
                    if cached.shape == (h_sub, self.token_dim):
                        W_plus[i] = cached.to(device=device, dtype=torch.float32)
                        needs_rollout_mask[i] = False

        # Batch teacher rollout for remaining samples.
        rollout_indices = needs_rollout_mask.nonzero(as_tuple=False).squeeze(-1)
        if rollout_indices.numel() > 0:
            x_prime_rollout = x_prime[rollout_indices]  # (B_r, obs+act)
            B_r = x_prime_rollout.shape[0]

            # Run full reverse chain: start from pure noise z_T ~ N(0, I)
            z = torch.randn(
                B_r, h_sub, self.token_dim, dtype=torch.float32, device=device
            )

            with torch.no_grad():
                for step in reversed(range(1, self.schedule.T + 1)):
                    t_batch = torch.full(
                        (B_r,), step, dtype=torch.long, device=device
                    )
                    # Teacher predicts the score at this timestep.
                    # _cont_loss uses the score convention; the teacher is a
                    # score network so we convert score -> noise for p_sample_step:
                    #   eps_pred = -sigma_t * score_pred
                    score_pred = teacher_score_fn(z, t_batch, x_prime_rollout, h_sub)

                    # Convert score to noise prediction for DDPM reverse step.
                    sigma_t_val = self._diffusion.sqrt_one_minus_alphas_bar[step].item()
                    noise_pred = -float(sigma_t_val) * score_pred

                    z = self._diffusion.p_sample_step(
                        score_net_output=noise_pred,
                        z_t=z,
                        t=step,
                        x=x_prime_rollout,
                    )
                # z is now the clean suffix sample W_+  (B_r, h_sub, token_dim)

            # Write back into W_plus at the correct indices.
            W_plus[rollout_indices] = z

            # Optionally cache the newly generated suffixes for later reuse.
            if replay_buffer is not None:
                for j, idx in enumerate(rollout_indices.tolist()):
                    try:
                        replay_buffer.add(
                            x_prime=x_prime[idx],
                            clean_suffix=z[j].detach().cpu(),
                        )
                    except Exception:
                        # Never let buffer I/O break training.
                        pass

        return W_plus

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BellmanDiffusionLoss("
            f"gamma={self.gamma}, "
            f"token_dim={self.token_dim}, "
            f"T={self.schedule.T})"
        )
