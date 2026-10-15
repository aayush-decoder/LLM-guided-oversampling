# What every module does

Package: `src/tiltgen`. Command: `tilt`.

Everything outside `methods/tilt`, `experiments/oracle.py`, `audits/theory_checks.py` and `core/evaluate/privacy.py` is vendored from palt3 (PROVENANCE.md). Those files keep their palt3 docstrings, which explain each correction.

## NEW in palt4: the TILT method (`methods/tilt/`)

| module | what it does |
|---|---|
| `utility.py` | `UtilityModel`, fitted on real D_train with 2-fold cross-fitting:<br>• utilities `fisher` (default; Fisher leverage, Prop. 2), `influence`, `epistemic` (bootstrap probes), `aleatoric` (ablation), `none`;<br>• the **validity floor**, from a non-linear class-balanced probe fitted out of fold;<br>• a Newton-method logistic regression, so \(H\) is exactly the Hessian the theory uses. |
| `tilt.py` | Standardise u to z; KL(q‖p) estimate; λ by bisection to the budget ε, capped by amplification A; ESS; Gumbel-top-k selection without replacement. |
| `privacy.py` | DCR gate against D_minor. Its threshold comes from an internal split of D_minor, so it never sees held-out rows. |
| `partial.py` | `PartialUtility`: the SMC twist. A gradient-boosted regressor of the full-row z on randomly masked rows (FUDGE for fields). |
| `smc.py` | `twisted_smc`: field-wise twisted SMC over any engine that can `complete`, with a clause parser, per-group resampling and a stats record. Exact for any positive twist (Prop. 3). |
| `method.py` | `TiltConfig` (extends V2Config), `PRESETS` (tilt_off, tilt_sir, tilt_smc, tilt_influence, tilt_epistemic, tilt_aleatoric, tilt_random, tilt_contrast, tilt_nogate), `TiltOversampler` (the v2 fine-tune, then a SIR or SMC sampling stage), and the `contrast` utility (sequence-level CFG). |

## NEW in palt4: evaluation, experiments, audits

| module | what it does |
|---|---|
| `core/evaluate/privacy.py` | Membership attack on the minority training rows (DOMIAS-style density ratio). `mia_auc` uses nearest-neighbour densities; `mia_auc_kde` uses KDEs. |
| `experiments/oracle.py` | The **no-LLM oracle study**: known Gaussian-mixture minority, exact p_min samples. Arms: threshold, oracle, SMOTE, and tilts at each budget. LR and XGBoost. |
| `experiments/grids.py` | + `thesis_main`, `utility`, `budget`, `sampler`, `validity`, `scale` (the Qwen ladder), `scarcity`, and TILT arms in `smoke`. |
| `experiments/report.py` | + `gain_vs_threshold` / `real_gain_vs_threshold`, paired per (dataset, seed). |
| `experiments/runner.py` | + kind `tilt`, the membership attack on every oversampler, and TILT diagnostics copied into metrics (`tilt_lambda`, `tilt_kl`, ...). |
| `audits/theory_checks.py` | Numerical checks of Lemma 1 and Props. 1–4 (THEORY.md). `tilt audit theory`. |
| `core/data/registry.py` | + the `natural` benchmark (10 tables) and `natural_fast`. OpenML default target via `_nat(id)`. |
| `core/data/fetch.py` | + `source["default_target"]`: renames OpenML's default target column to `target`. |
| `core/llm/zoo.py` | + qwen3-14b, qwen2.5-14b, qwen2.5-32b. Qwen3.5 and MoE checkpoints are rejected, with the reason. |
| `core/baselines/classical.py` | + `borderline_smote`, `svm_smote` (boundary-emphasis baselines). |
| `cli.py` | + `tilt oracle`, `tilt audit theory`, `tilt train --method tilt --preset ...`. |

## Vendored from palt3 (unchanged, apart from the rename)

| area | modules |
|---|---|
| data | `core/data/`: registry, fetch, protocol (splits, q-reduction, validation fold), kinds (discrete / ordinal) |
| LLM | `core/llm/`: zoo (model switch), hardware (planner), engine (the only LLM code: finetune / complete / score) |
| baselines | `core/baselines/`: classical (SMOTE family), deep (CTGAN/TVAE/TabDDPM/ARF), curation (CLLM-style) |
| evaluation | `core/evaluate/`: downstream (XGB + LR/RF/MLP), quality, diversity, stats, covariates |
| bookkeeping | `core/runs.py`, `core/serialize.py`, `core/viz.py` |
| methods | `methods/pipeline.py`, `v1_paper` (ImbLLM as published), `v2_corrected` (ImbLLM corrected), `llm_baselines` |
| audits | `audits/`: math_checks, paper_tables, data_checks, contamination |

## Data flow

```
raw csv -> protocol -> Splits (train_major, train_minor, minor_heldout, val, test)
                           |
          v2 corpus -> engine.finetune (cached per dataset x seed)
                           |
     TILT  ---------------+--------------------------------------------+
     SIR: v2 pool (x4) -> validity + DCR gate -> z(u) -> λ(ε, A)        |
                       -> Gumbel-top-k ∝ e^{λz}                          |
     SMC: pilot -> λ, twist ẑ(x_S) -> field-wise twisted SMC -> keep    |
                           |                                            |
     D_major ∪ D^_minor (and ∪ D_minor) -> XGB / LR / RF / MLP -> F1, AUPRC, ...
     + quality (held-out) + mia_auc + cost_* + cov_* -> metrics.json
     report: gain_vs_threshold, Friedman/Nemenyi, Holm-Wilcoxon, covariates, cost
```
