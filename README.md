# Bellman Path Diffusion for Direct Off-Policy Trajectory Generation

Reference implementation of **Bellman Path Diffusion (BPD)**, a trajectory-space extension of the probability-path Bellman construction from [Temporal Difference Flows](https://arxiv.org/abs/2503.09817).

## Overview

BPD directly generates complete reward-bearing trajectories under a fixed evaluation policy from one-step offline transition data, without ever running a sequential world model at inference. The central identity is:

$$m_{h,t}^{\pi_e} = (1-\gamma)\,m_{h,t}^{\text{stop}} + \gamma\,m_{h,t}^{\text{cont}}, \qquad t \in [0,T]$$

Both branches share a common Gaussian source at $t=T$. The **stop branch** score is analytic; the **continuation branch** score bootstraps from a frozen shorter-horizon teacher at the same noise level. A single reverse diffusion from $\mathcal{N}(0, I_{Hd})$ generates a complete padded trajectory tensor $W_H \sim \mathbb{M}_H^{\pi_e}$.

**Key properties:**
- Off-policy: conditions on single-step logged transitions, no importance weights required (Assumption 1)
- Geometric survival implements discounting — the undiscounted reward sum $\sum_{t < L_H} R_t$ is an unbiased estimator of $\mathbb{E}_{\pi_e}[\sum_{t=0}^{H-1} \gamma^t R_t]$ (Proposition 1 / Eq. 15)
- Stagewise training: horizon $h$ uses frozen horizon $h-1$ as teacher; one reverse SDE at horizon $H$ at inference
- Replay equivalence: cache clean suffixes once, corrupt on-the-fly — population-equivalent to partial reverse SDE (Proposition 2)

## Project Structure

```
bpd/
├── core/
│   ├── path.py          # stop_map, push_map, path_length, compute_return (Eq. 4-5, 14-15)
│   ├── branches.py      # StopBranch, ContinueBranch score targets (Eq. 41-47)
│   └── objectives.py    # BellmanDiffusionLoss L_h(θ; θ̄) (Eq. 48)
├── data/
│   ├── dataset.py       # TransitionDataset, D4RLTransitionDataset
│   ├── normalizer.py    # LimitsNormalizer, GaussianNormalizer
│   └── replay.py        # SuffixReplayBuffer (Section 6 / Proposition 2)
├── models/
│   ├── diffusion.py     # DDPMSchedule, BlockwiseDiffusion (Eq. 22-25)
│   ├── schedule.py      # VPSchedule (continuous-time VP-SDE)
│   └── score_net.py     # TrajectoryScoreNet (DiT-style Transformer, Remark 3)
├── training/
│   ├── ema.py           # EMA target network (Appendix D)
│   └── trainer.py       # BellmanPathDiffusionTrainer (Algorithm 1)
├── evaluation/
│   └── ope.py           # OPEEvaluator, Ĵ_H(π) estimator (Algorithm 2, Eq. 59)
└── utils/
    ├── arrays.py        # DataBatch, tensor utilities
    ├── logger.py        # Logging (console + file + optional W&B)
    ├── serialization.py # Checkpoint save/load
    └── timer.py         # Wall-clock timing
configs/
├── base.yaml
├── hopper_medium.yaml
└── halfcheetah_medium.yaml
scripts/
├── train.py             # Stagewise training entry point
└── evaluate.py          # OPE evaluation entry point
tests/
├── test_path.py         # Path algebra, length law, value identity
├── test_diffusion.py    # Blockwise diffusion, score targets
├── test_objectives.py   # Exact stop/continuation targets and loss
└── test_training.py     # Stagewise trainer, replay, decoder integration
```

## Installation

```bash
# Core dependencies
pip install torch>=2.0.0 numpy einops pyyaml tqdm scipy

# Optional: D4RL datasets (MuJoCo license required)
pip install d4rl gym

# Optional: W&B logging
pip install wandb

# Install package
pip install .
```

## Quick Start

### Training

```bash
# Train on Hopper (D4RL)
python scripts/train.py \
    --config configs/hopper_medium.yaml \
    --gamma 0.99 \
    --max_horizon 8 \
    --steps_per_horizon 100000 \
    --seed 42

# Override individual hyperparameters
python scripts/train.py \
    --env halfcheetah-medium-v2 \
    --gamma 0.99 \
    --max_horizon 16 \
    --batch_size 512 \
    --lr 3e-4 \
    --device cuda
```

### Evaluation (OPE)

```bash
python scripts/evaluate.py \
    --checkpoint results/hopper/ckpt_H8.pt \
    --env hopper-medium-v2 \
    --n_trajectories 1000 \
    --seed 42
```

### Programmatic Usage

```python
import torch
from bpd.models.diffusion import DDPMSchedule, BlockwiseDiffusion
from bpd.models.score_net import TrajectoryScoreNet
from bpd.core.objectives import BellmanDiffusionLoss
from bpd.data.replay import SuffixReplayBuffer

# Setup
obs_dim, act_dim = 11, 3
token_dim = 2 + obs_dim + act_dim  # reward + next_obs + next_action + injective-phi flag
gamma = 0.99
H = 8  # max horizon

schedule = DDPMSchedule.make_cosine(T=1000)
diffusion = BlockwiseDiffusion(schedule, token_dim)
score_net = TrajectoryScoreNet(
    obs_dim=obs_dim,
    act_dim=act_dim,
    token_dim=token_dim,
    max_horizon=H,
)

loss_fn = BellmanDiffusionLoss(
    gamma=gamma,
    schedule=schedule,
    token_dim=token_dim,
    diffusion=diffusion,
)

# Teacher score function (frozen EMA at horizon h-1)
@torch.no_grad()
def teacher_noise_fn(z, t, x, h_sub):
    return score_net(z, t, x, h_sub)

# Training step
replay_buffer = SuffixReplayBuffer(max_size=50000, suffix_horizon=H-1, token_dim=token_dim)
next_action = evaluation_policy(batch.next_state)  # a' ~ pi_e(. | s')
loss = loss_fn(
    score_net,
    teacher_noise_fn,
    batch,
    h=H,
    next_action=next_action,
    replay_buffer=replay_buffer,
)
loss.backward()
```

## Algorithm Summary

### Training (Algorithm 1)

For $h = 1, \ldots, H$:
1. Freeze teacher $s_{\bar\theta, h-1}$ (EMA or exact copy)
2. Initialise suffix replay buffer $\mathcal{R}_{h-1}$
3. For each gradient step:
   - Sample $(s, a, r, s') \sim \mathcal{D}$, draw $a' \sim \pi_e(\cdot|s')$, form token $y = (r, s', a')$
   - Sample $t \sim p_T$, draw branch $C \sim \text{Bernoulli}(\gamma)$
   - **Stop branch** ($C=0$ or $h=1$): predict the sampled noise for the full stopped path.
   - **Continue branch** ($C=1$, $h>1$): retrieve $w_+ \sim \hat{\mathbb{M}}_{\bar\theta,h-1}(\cdot|x')$ using the exact `(s', a')` key; predict `[head noise; frozen teacher suffix noise]` at the same $t$.
   - Loss: $\lambda(t)\|\epsilon_\theta(z_t,t|x,h)-\epsilon_{\mathrm{target}}\|_2^2$. The branch draw supplies the only factor of $\gamma$.

### Generation (Algorithm 2)

1. Sample $s_0 \sim \mu_0$, $a_0 \sim \pi_e(\cdot|s_0)$
2. Sample $z_T \sim \mathcal{N}(0, I_{Hd})$
3. Run reverse SDE $T \to 0$ conditioned on $x_0 = (s_0, a_0)$
4. Decode $W_H$, remove padding; return trajectory

### OPE Estimator (Eq. 59)

$$\hat{J}_H(\pi_e) = \frac{1}{M}\sum_{m=1}^M \sum_{t=0}^{L_H(W_H^{(m)})-1} R_t^{(m)}$$

No $\gamma^t$ factor — geometric stopping already implements discounting.

## Theoretical Guarantees

| Result | Statement |
|--------|-----------|
| **Theorem 1** (Trajectory Bellman probability path) | $m_{h,t}^{\pi_e} = (1-\gamma)m_{h,t}^{\text{stop}} + \gamma m_{h,t}^{\text{cont}}$ for all $t \in [0,T]$ (Eq. 29) |
| **Lemma 1** (Mixture-score regression) | Population minimizer of branch regression = $\nabla_z \log m_{h,t}^{\pi_e}$ (Eq. 35) |
| **Theorem 2** (Population score backup) | Exact teacher ⟹ student recovers exact score (Eq. 53) |
| **Theorem 3** (Stagewise exact recovery) | Induction recovers $W_H \sim \mathbb{M}_H^{\pi_e}$ exactly (Eq. 54) |
| **Proposition 2** (Replay equivalence) | Clean suffix + forward corruption $\equiv$ partial reverse SDE (population) |
| **Lemma 2** (Full-law stability) | $\|\mathcal{B}_h^\pi Q - \mathcal{B}_h^\pi Q'\|_{\text{TV},\infty} \leq \gamma \|Q - Q'\|_{\text{TV},\infty}$ (Eq. 56) |
| **Theorem 4** (TV error bound) | $\|\hat{\mathbb{M}}_h - \mathbb{M}_h^{\pi_e}\|_{\text{TV},\infty} \leq \sum_{j=1}^h \gamma^{h-j}\varepsilon_j^{\text{TV}}$ (Eq. 57) |
| **Theorem 5** (Return contraction) | $\|\mathcal{B}_h^\pi Q - \mathcal{B}_h^\pi Q'\|_{G,h} \leq \gamma \|Q - Q'\|_{G,h-1}$ (Eq. 65) |
| **Theorem 6** (Return error bound) | $\delta_H^G \leq \sum_{h=1}^H \gamma^{H-h}\varepsilon_h^G$ (Eq. 68) |
| **Corollary 1** (OPE error) | $|\bar{J}_H(\pi_e) - J_H(\pi_e)| \leq \delta_H^G$ (Eq. 61) |

## Noise Network Architecture

A single `TrajectoryScoreNet` handles all horizons $h \in \{1, \ldots, H\}$ via:
- **Token projection**: each block $z_j \in \mathbb{R}^{d_\text{tok}} \to e_j \in \mathbb{R}^D$
- **Time embedding**: sinusoidal $\to$ MLP (DiT-style, Peebles & Xie 2023)
- **Horizon embedding**: MLP over integer $h$
- **Conditioning**: linear projection of $x = (s, a)$
- **AdaLN-Zero Transformer blocks**: adaptive layer norm modulated by time + horizon + state-action conditioning
- **Variable length**: each homogeneous stage passes exactly the first $h$ blocks, equivalent to masking a padded shared-H tensor
- **Output projection**: $e_j \to \epsilon_j \in \mathbb{R}^{d_\text{tok}}$; the score is $-\epsilon_j/\sigma_t$

See [`docs/IMPLEMENTATION_AUDIT.md`](docs/IMPLEMENTATION_AUDIT.md) for the paper-to-code map, reference implementations consulted, and known approximation boundaries.

## Citation

```bibtex
@article{bellman_path_diffusion_2025,
  title   = {Bellman Path Diffusion for Direct Off-Policy Trajectory Generation},
  year    = {2025},
}
```

## References

- Farebrother et al. (2025). *Temporal Difference Flows*. ICML 2025.
- Ho et al. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
- Song et al. (2021). *Score-Based Generative Modeling through SDEs*. ICLR 2021.
- Peebles & Xie (2023). *Scalable Diffusion Models with Transformers (DiT)*. ICCV 2023.
- Janner et al. (2022). *Planning with Diffusion*. ICML 2022.
- Ying et al. (2026). *Temporal Difference Learning for Diffusion Models (TD²-DD)*. ICML 2026.
