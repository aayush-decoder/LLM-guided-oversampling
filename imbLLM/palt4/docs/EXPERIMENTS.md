# Experiments

Every experiment is one command. Each has a stated hypothesis and a falsification condition. The order below is the order to run them in: each step decides whether the next is worth paying for.

## Protocol

- **Tables.**
  - `natural`: 10 naturally imbalanced OpenML tables, IR 5.5–67 (DATASETS.md).
  - `natural_fast`: pc1, kc1_nat, wilt, for development.
  - The ImbLLM `paper` set under its own protocol (`replication.yaml`), for comparability with palt3.
- **Splits.**
  - Stratified 80/20 train/test.
  - A 10% real validation fold (protocol `clean`) for every threshold.
  - q = 0.8: the generator sees 80% of the real minority. The other 20% is the held-out reference for fidelity and the membership attack.
- **Generation.** |largest class| minority rows, as in ImbLLM.
- **Learners.** XGBoost (the unprefixed keys), plus LR, RF and MLP (`lr_*`, `rf_*`, `mlp_*`).
- **Seeds.**
  - 10 on tables under 5k rows; at least 5 on bank_full and mammography.
  - Every paired test is over datasets. With N datasets the smallest possible Wilcoxon p is 2·0.5^N, so at least 6 datasets are needed.
- **Headline metric.** `gain_vs_threshold = f1_tuned(method) − f1(threshold_only)`, paired per (dataset, seed); both thresholds are tuned on the same real validation fold. Also reported:
  - `real_gain_vs_threshold` (the real minority rows kept);
  - AUPRC and balanced accuracy;
  - held-out close-probability and coverage;
  - `mia_auc`;
  - cost.

## E0. Oracle study, no LLM (`tilt oracle`), CPU, minutes

Question: when the generator is *perfect*, does tilting beat untilted oversampling, and does either beat the tuned threshold?

| hypothesis | falsified if |
|---|---|
| **H1** (Prop. 1): `oracle` ≈ `threshold_only` as n_min grows | `oracle` keeps a gain at n_min = 100+ |
| **H2** (Prop. 2): `tilt_fisher` > `oracle` at small n_min, and the gap shrinks as n_min grows | `tilt_fisher` ≤ `oracle` at n_min = 20 across budgets |
| **H4**: `tilt_aleatoric` < `tilt_fisher` | aleatoric ≥ fisher |

Command: `tilt oracle --n-min 20 50 100 --seeds 30 --learners lr xgb --budgets 0.1 0.25 0.5`

Then repeat with `--unimodal`, and with `--ir 50`.

**First look** (3 seeds, n_min = 20, LR, ε = 0.25; not conclusive):

| arm | gain vs threshold |
|---|---|
| oracle | +0.024 |
| fisher | +0.018 |
| aleatoric | +0.012 |
| epistemic | +0.008 |
| influence | +0.005 |

Perfect untilted oversampling *did* beat the tuned threshold at this sample size (finite-sample regularisation). No tilt beat it yet. **Run the full E0 before anything else.**

**If H2 fails:** the thesis becomes "why utility guidance does not beat a perfect generator, and when it does". Sweep dimension, separation, IR and budget to find the regime. It remains a publishable, honest result.

## E1. `thesis_main`: the headline table

Arms:
- references: `original`, `imbalance`, `threshold_only`, `prior_shift`, `scale_pos_weight`;
- classical: `smote_nc`, `borderline_smote`;
- LLM baselines: `imbllm_paper`, `v2_recommended`, `v2_recommended+cllm`;
- TILT: `tilt_sir`, `tilt_smc`.

**H3:** `tilt_sir` has `gain_vs_threshold` > 0 on most tables, and more than `v2_recommended` (Holm-corrected Wilcoxon over datasets). Report each learner separately.

Commands:
- `tilt run thesis_main --config h100_main --resume`
- `tilt report thesis_main --metric gain_vs_threshold`

## E2. `utility`: which u?

Arms: fisher, influence, epistemic, aleatoric, none, contrast.

- **H4:** fisher ≥ epistemic > none > aleatoric.
- Check with the covariate table (`cov_minority_modes`, `cov_class_overlap`): aleatoric should hurt most where overlap is high.

## E3. `budget`: the utility–fidelity–privacy frontier

ε ∈ {0, .05, .1, .25, .5, 1, 2}.

- **H5:** gain rises and then falls in ε, as Prop. 2's saturation predicts.
- Held-out close-probability falls monotonically.
- `mia_auc` rises.

Plot all three against the realised `tilt_kl`. **This figure is the thesis's main picture.**

## E4. `sampler`: SIR vs twisted SMC at equal compute

SIR pool ×2/×4/×8 vs SMC particles 4/8/16. Compare gain, `tilt_ess` and `cost_*`.

**H6:** SMC reaches SIR×8 quality at lower generated-token cost on tables with M ≥ 15 (early pruning pays off with more fields).

## E5. `validity`: the label-noise defence

Arms: floor on/off, floor at 5%/20%, DCR gate on/off.

**H7:** without the floor, gain falls and the false-positive rate rises, most on high-overlap tables.

## E6. `scale`: does a bigger LLM help TILT, or only the generator?

Qwen3-0.6B → 14B, plus Qwen2.5-32B, for v2 vs TILT-SIR. H100 only (COMPUTE.md).

**H8:** the TILT − v2 gap is roughly constant across scale. The utility is model-agnostic; scale improves p_gen.

## E7. `scarcity`: q ∈ {.05, .1, .2, .5}

Here the guidance should matter most. This mirrors E0 on real tables.

## E8. Replication protocol (`--config replication`)

`thesis_main` on the ImbLLM paper tables with q = 0.2, so numbers line up with palt3's ImbLLM replication.

## E9. Contamination

`tilt audit contamination --model qwen3-1.7b` before any claim on adult, german or bank. Rows the backbone memorised would inflate every LLM arm.

## Figures and tables to produce

1. E0: gain vs n_min per arm (lines with 95% CI). Proposition 1 and 2 in one picture.
2. E1: table of `gain_vs_threshold` (mean ± sd), datasets × methods, per learner, with a critical-difference diagram.
3. E3: the frontier (gain, fidelity, `mia_auc` vs KL).
4. E4: gain vs generated tokens (SIR vs SMC).
5. E6: gain vs parameters (log x).
6. Covariates: Spearman of gain vs `cov_*`.
