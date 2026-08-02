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
| Temporal Difference Flows, ICML 2025 | [paper](https://arxiv.org/abs/2503.09817) and its published pseudocode | probability-path Bellman branching and frozen target field | no official public code repository was discoverable during this audit |

## Approximation and validation boundaries

- The exact recovery theorem assumes realizability, population optimization,
  an exact teacher, and exact reverse integration. The implementation uses a
  finite network, SGD, EMA snapshots, and a finite DDPM solver.
- Standard finite DDPM schedules make the terminal marginal numerically close
  to, rather than symbolically equal to, `N(0,I)`.
- Clean-suffix replay is exact only conditional on the same float32 `(s',a')`.
  Rounded or nearest-neighbor replay is disabled by default. A finite cached
  sample reused across updates remains a finite-sample approximation.
- Continuous zero-padding is decoded by distance to the reserved padding
  embedding. This is a practical decoder, not a proof of exact inversion of
  `Phi_h`; the paper itself identifies exact discrete decoding as nontrivial.
- Coverage of every evaluation-policy state-action pair cannot be established
  from software alone. Training/evaluation must report a coverage diagnostic
  for each dataset-policy pair.
- The included tests verify equations, shapes, gradients, replay keys, and a
  two-stage CPU smoke run. They are not benchmark results. A paper claim of
  empirical OPE accuracy requires full seeded experiments and baselines.
