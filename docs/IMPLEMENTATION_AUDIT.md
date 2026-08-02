# Implementation audit

This document records how `paper.tex` maps to executable code. It also makes
the practical approximations explicit; passing unit tests is not presented as
evidence for the paper's population assumptions or empirical performance.

## Paper-to-code map

| Paper object | Implementation | Verification |
|---|---|---|
| Token `y=(r,s',a')`, stop/push maps, length and return identities | `bpd/core/path.py` | `tests/test_path.py` |
| Absorbing state-action pair with zero future reward | `bpd/data/dataset.py` redirects terminal successors and adds an absorbing self-transition; trainer forces absorbing next-actions to zero | `test_dataset_adds_absorbing_zero_reward_self_transition` |
| Independent blockwise corruption, Eqs. (22)--(25) | `bpd/models/diffusion.py` | `tests/test_diffusion.py` |
| Stop target, Eqs. (41)--(43), DDPM Appendix C | `bpd/core/branches.py::StopBranch` | `test_stop_target_is_sampled_noise` |
| Continuation target, Eqs. (44)--(47), DDPM Appendix C | `bpd/core/branches.py::ContinueBranch` | `test_continuation_is_head_noise_plus_unscaled_teacher_noise` |
| Mixture objective, Eq. (48) | `bpd/core/objectives.py::BellmanDiffusionLoss` | `tests/test_objectives.py` |
| Exact `(s',a')` clean-suffix replay, Section 6 | `bpd/data/replay.py` | `test_replay_default_key_is_exact` |
| Stagewise frozen teachers, Algorithm 1 | `bpd/training/trainer.py` | `test_two_horizon_training_smoke` |
| EMA practical variant, Appendix D | `bpd/training/ema.py` | exercised by the stagewise smoke test |
| One horizon-H reverse chain, Algorithm 2 | `bpd/models/diffusion.py::p_sample_loop`, `bpd/evaluation/ope.py` | diffusion and decoder tests |
| Undiscounted generated-path reward sum, Eq. (59) | `bpd/evaluation/ope.py` | path/value tests and evaluator tests |

The executable model uses the paper's Appendix C epsilon parameterization.
For a VP/DDPM marginal, `score = -epsilon / sigma_t`. Therefore the stop target
is sampled Gaussian noise, and the continuation target is the concatenation of
head noise and frozen teacher suffix noise. `gamma` is used once to draw the
branch and is not multiplied into the teacher target.

## External implementations consulted

The repositories were shallow-cloned under the ignored local `references/`
directory. No reference repository is vendored into this project.

| Work | Repository and inspected commit | Pattern used | License note |
|---|---|---|---|
| Diffuser, ICML 2022 | [jannerm/diffuser](https://github.com/jannerm/diffuser), `7ea422860cc0106e5ca5949d980f04b799d5462c` | separation of `q_sample`, posterior/reverse sampling, EMA training lifecycle, field normalization boundaries | MIT |
| Score-SDE, ICLR 2021 Oral | [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch), `cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44` | score/epsilon conversion and stop-gradient reverse-SDE organization | Apache-2.0 |
| DiT, ICCV 2023 | [facebookresearch/DiT](https://github.com/facebookresearch/DiT), `ed81ce2229091fd4ecc9a223645f95cf379d582b` | adaLN-Zero gating and zero-initialized final projection | CC-BY-NC; consulted as an architectural reference, not vendored |
| DDPM, NeurIPS 2020 | [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion), `1e0dceb3b3495bbe19116a5e1b3596cd0706c543` | original discrete DDPM forward/posterior parameterization (`ho2020ddpm`) | MIT; consulted, not vendored |
| Gamma-Models, NeurIPS 2020 | [JannerM/gamma-models](https://github.com/JannerM/gamma-models), `bb23e06753717a9255065355b5f6ab77278f3305` | geometric-horizon / discounted-future sampling and geometric survival (`janner2020gamma`), the predecessor of the geometric stopping in Eq. 14-15 | MIT; consulted, not vendored |
| Temporal Difference Flows, ICML 2025 | [paper](https://arxiv.org/abs/2503.09817) and its published pseudocode | probability-path Bellman branching and frozen target field | no official public code repository was discoverable during this audit (arXiv, OpenReview, ICML v267 proceedings, and author/org GitHub all checked) |

## Approximation and validation boundaries

- The exact recovery theorem assumes realizability, population optimization,
  an exact teacher, and exact reverse integration. The implementation uses a
  finite network, SGD, EMA snapshots, and a finite DDPM solver.
- Source boundary `alpha_T = 0, sigma_T = 1` (paper.tex L409) is enforced
  **exactly**: `DDPMSchedule` applies a zero-terminal-SNR rescaling of
  `sqrt(alphabar_t)` (Lin et al. 2024, "Common Diffusion Noise Schedules and
  Sample Steps are Flawed", Algorithm 1) so the terminal marginal is
  symbolically `N(0,I)`, not merely numerically close. Enabled by default via
  `zero_terminal_snr=True`.
- **v-prediction is the default parameterization** (`BlockwiseDiffusion(...,
  prediction_type="v")`), which is what makes the exact `alpha_T = 0` source
  usable: x0 recovery uses `x0 = alpha_t*z_t - sigma_t*v` (finite at the
  source, `x0 = -v`), avoiding the singular epsilon recovery `x0 =
  (z_t - sigma_t*eps)/alpha_t` whose `1/alpha_T` factor amplifies terminal
  error (Lin et al. 2024 pair rescaling with v-prediction for exactly this
  reason; Salimans & Ho 2022 for the v-objective). Training targets, the frozen
  teacher, and the reverse sampler all share the parameterization; `"epsilon"`
  remains selectable. Both are the same score field up to an affine change of
  basis, so the Bellman branch composition (stop-gradient teacher suffix, no
  gamma multiplier) is unchanged.
- The token map `phi` is now **injective by construction** (paper.tex L384):
  each token carries a trailing flag coordinate, `+1` for a real token and
  `-1` for the padding token `phi(bottom) = (0,...,0,-1)`. Decoding is the
  exact inverse `Phi_h^{-1}` — a slot is padding iff its flag coordinate is
  negative — replacing the earlier `||token|| < threshold` heuristic under
  which a near-zero real transition could be misread as padding. This makes
  `d_tok = 2 + obs_dim + act_dim`.
- Clean-suffix replay is exact only conditional on the same float32 `(s',a')`.
  Rounded or nearest-neighbor replay is disabled by default. A finite cached
  sample reused across updates remains a finite-sample approximation.
- Data/infra correctness fixes: the D4RL loader now calls
  `d4rl.qlearning_dataset(env)` (which constructs `next_observations`) and
  routes downloads to a repo-local, gitignored `data/d4rl` via
  `D4RL_DATASET_DIR`, never `$HOME/.d4rl`. Checkpoints load with
  `weights_only=False` (PyTorch >= 2.6) and directory loads prefer `final.pt`
  (the only checkpoint bundling full config + normalizer). `encode_token` is
  device-safe for CUDA datasets. The bootstrap CI RNG is seeded from the run
  seed. The YAML/CLI merge detects explicitly-passed flags from `argv` (a CLI
  value equal to the argparse default is no longer overridden by YAML).
  `corollary1_error_bound` now documents its argument as the return-seminorm
  (G-seminorm) stage error epsilon_h^G (Eq. 66), not a score-matching residual.
- Initial-state distribution mu_0: the D4RL loader computes episode-start
  observations from the raw episode boundaries (`terminals | timeouts`) BEFORE
  the qlearning conversion and returns them as `episode_start_observations`.
  qlearning_dataset drops timeouts, so deriving mu_0 from its output would
  collapse the start pool to a single state on time-limit-terminated
  locomotion data; the evaluator now consumes the explicit pool.
- The diffusion parameterization is self-describing: `prediction_type` is a CLI
  flag, is saved in the training config/checkpoint, and is read back at
  evaluation, so an epsilon checkpoint is never silently loaded as a v model.
- v-prediction is now documented in the paper (Appendix, "Velocity
  Parameterization at the Exact Source"): the v<->epsilon<->score transforms,
  the branch v-targets, the frozen-teacher recursion in the v basis, and the
  exact-source rationale, with the Lin et al. 2024 and Salimans & Ho 2022
  citations added to the bibliography.
- Coverage of every evaluation-policy state-action pair cannot be established
  from software alone. Training/evaluation must report a coverage diagnostic
  for each dataset-policy pair.
- Deferred (not yet implemented, do not affect the core recursion): a
  target-policy checkpoint loader in the CLI (the API already accepts any
  policy callable; the CLI exposes only `dataset` and `random`), a dedicated
  absorbing-state conditioning flag, an executable coverage diagnostic, and
  true stage/step training resume. `gym>=0.26` is declared but the rollout code
  uses the older `reset/step/env.seed` API.
- The included tests verify equations, shapes, gradients, replay keys, the
  v-prediction source stability, and a two-stage CPU smoke run. They are not
  benchmark results. A paper claim of empirical OPE accuracy requires full
  seeded D4RL experiments, baselines, and multiple seeds.
