# palt4 command reference

Every command of the `tilt` CLI, every experiment, every test marker and every Make target.

- **Installation:** [SETUP.md](SETUP.md).
- **What the experiments test:** [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
- **Running commands:** from `palt4/`, as `tilt <command>` or `python -m tiltgen <command>`. `tilt <command> -h` prints the flags of any command.
- **Downloads:** nothing downloads until a command needs it.
  - `tilt data fetch` downloads tables.
  - `tilt pull`, or the first LLM run, downloads model weights.

## Index

| group | commands |
|---|---|
| [1 Data](#1-data) | `data list` · `data fetch` · `data verify` · `data inspect` |
| [2 Figures](#2-figures) | `viz data` · `viz synth` · `viz report` |
| [3 LLM switch](#3-llm-switch) | `models` · `plan` · `pull` · `hw` · `where-llm` |
| [4 Checks and tests](#4-checks-and-tests) | `pytest` markers · `audit theory` / `math` / `paper` / `data` / `all` / `contamination` · `smoke` |
| [5 Oracle study](#5-oracle-study-no-llm) | `oracle` |
| [6 One method](#6-one-method-train-and-infer) | `train` · `infer` |
| [7 Experiments](#7-experiment-grids) | `run <experiment>` (23 grids) |
| [8 Reports](#8-reports) | `report` |
| [9 Housekeeping](#9-housekeeping) | `clean` |
| [10 Make targets](#10-make-shortcuts) | all 51 targets |

---

## 1. Data

| command | what it does | flags |
|---|---|---|
| `tilt data list` | registry: 41 entries, source, groups, whether downloaded | – |
| `tilt data fetch` | download raw tables into `data/raw/` | `--datasets` (name, comma list or group; default `open`) · `--from DIR` (import csvs from another folder, offline) · `--force` (re-download) |
| `tilt data verify` | cleaned shape vs the registry's expected N / M / K, per table | `--datasets` (default `all`) |
| `tilt data inspect` | splits, class counts, imbalance ratio, discrete / ordinal columns for one table | `--dataset` (required) · `--seed` · `--q` · `--protocol clean\|paper` · `--export` (write the splits as csv to `data/processed/`) |

**Dataset groups**, usable wherever a flag takes `--datasets`:

| group | members |
|---|---|
| `natural` | the 10 thesis tables |
| `natural_fast` | pc1, kc1_nat, wilt |
| `paper` | the 10 ImbLLM tables |
| `predllm` | the 20 Pred-LLM tables |
| `fast` | australian, fuel, blood |
| `open` | no credentials needed |
| `kaggle_only` | credit_card, gem_price |
| `leak_free` | cardio_noleak |
| `smoke` | synthetic |
| `all` | every registry entry |

A comma list also works, for example `pc1,wilt`.

```bash
tilt data list
```

```bash
tilt data fetch --datasets natural
```

```bash
tilt data fetch --datasets all --from ../palt3/data/raw
```

```bash
tilt data fetch --datasets credit_card,gem_price --force
```

```bash
tilt data verify --datasets natural
```

```bash
tilt data inspect --dataset pc1 --q 0.8 --protocol clean --export
```

## 2. Figures

| command | what it draws | flags |
|---|---|---|
| `tilt viz data` | per table: raw vs clean, protocol funnel, class balance, minority geometry → `data/figures/<table>/` | `--datasets` (default `fast`) · `--seed` · `--q` · `--protocol` |
| `tilt viz synth` | one run: real vs synthetic marginals, PCA, DCR | `--run runs/<exp>/<table>/<method>/seed<k>` (required) |
| `tilt viz report` | the figure of one experiment's report | `EXPERIMENT` (positional) · `--metric` (default `f1`) |

```bash
tilt viz data --datasets natural_fast --q 0.8
```

```bash
tilt viz synth --run runs/thesis_main/pc1/tilt_sir/seed0
```

```bash
tilt viz report thesis_main --metric gain_vs_threshold
```

## 3. LLM switch

| command | what it does | flags |
|---|---|---|
| `tilt models` | every compatible backbone × every hardware profile (mode / precision, or "no"), plus the rejected families with reasons | `--seq-len` (default 256) |
| `tilt plan` | how one backbone runs on one device: mode, precision, micro-batch, accumulation, VRAM estimate, or a refusal with the reason | `--model` (required) · `--hardware` (default `auto`) · `--mode auto\|full\|lora\|qlora` · `--seq-len` |
| `tilt pull` | pre-download weights into `HF_HOME` | `--model KEY [KEY ...]` |
| `tilt hw` | detected device, plus the profiles in `configs/hardware/` | – |
| `tilt where-llm` | every place the LLM is called | `--grep` (also list the call sites found in the source) |

**Model keys** (Qwen ladder):

| machine | keys |
|---|---|
| laptop | `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b` |
| H100 | `qwen3-8b`, `qwen3-14b`, `qwen2.5-32b` |

Also registered: `qwen2.5-0.5b` / `1.5b` / `3b` / `7b` / `14b`, `smollm2-*`, `gpt2*`, `distilgpt2`, `llama3.x` (gated), and `tiny-random` (offline tests).

**Hardware profiles:** `rtx4050_6gb`, `h100_80gb`, `a100_40gb`, `t4_16gb`, `p100_16gb`, `cpu`, `auto`.

```bash
tilt models
```

```bash
tilt plan --model qwen3-1.7b --hardware rtx4050_6gb
```

```bash
tilt plan --model qwen2.5-32b --hardware h100_80gb --mode qlora
```

```bash
tilt pull --model qwen3-0.6b qwen3-1.7b
```

```bash
tilt hw
```

```bash
tilt where-llm --grep
```

## 4. Checks and tests

### pytest

| command | what runs | needs |
|---|---|---|
| `pytest -q -m "not llm and not slow"` | the 104 fast tests: data, protocol, serialisation, evaluation, TILT utilities / budget / SMC / oversampler, membership attack, theory, kinds | no torch, no network |
| `pytest -q -m "not slow"` | + the `llm` tests: end-to-end ImbLLM v1/v2 and TILT SIR/SMC on the offline tiny backbone, model-cache hits | torch |
| `pytest -q` | + the `slow` tests: a real backbone from the Hub | torch + network |
| `pytest -q tests/test_tilt.py` | only the TILT tests | – |
| `pytest -q -k smc` | tests whose name matches | – |

### Audits

| command | what it checks | flags |
|---|---|---|
| `tilt audit theory` | the thesis propositions, numerically: Lemma 1 (Gibbs), Prop. 1 (population equivalence), Prop. 2 (first-order law), Prop. 3 (SMC exactness), Prop. 4 (amplification cap), budget monotonicity | `-v` (print the numbers) |
| `tilt audit math` | palt3's math checks on shipped code: ⅔ covariance, geometry, τ-bridge, entropy identity, decomposition, permutation, close probability | `-v` |
| `tilt audit paper` | ImbLLM's published tables: transcription, Friedman / Nemenyi, Wilcoxon + Holm, the Table III decomposition | `-v` |
| `tilt audit data` | downloaded tables: shape, train/test duplicates, single-feature leaks, protocol disjointness | `--datasets` (default `all`) |
| `tilt audit all` | theory + math + paper + data (no LLM) | `--datasets` · `-v` |
| `tilt audit contamination` | memorisation probes on a **pre-trained** backbone: header, row, feature and first-cell completion vs baselines → `runs/_audits/` | `--model` (default `distilgpt2`) · `--datasets` (default `adult,german,bank,compas`) · `--trials` (default 25) · `--hardware` |

```bash
tilt audit theory -v
```

```bash
tilt audit all --datasets natural
```

```bash
tilt audit contamination --model qwen3-1.7b --datasets adult,german,bank,compas --trials 50
```

### Smoke test

| command | what it does | flags |
|---|---|---|
| `tilt smoke` | every code path on the offline tiny-random backbone: references, SMOTE-NC, ImbLLM v1, v2 recommended / full, TapTap-style, TILT-SIR, TILT-SMC. Checks every artefact and prints `SMOKE PASS` | `--datasets` (default `smoke`) · `--hardware` · `--strict` (stop at the first failure, with the traceback) |

```bash
tilt smoke
```

```bash
tilt smoke --hardware cpu --strict
```

## 5. Oracle study (no LLM)

| command | what it does | flags |
|---|---|---|
| `tilt oracle` | E0: a known Gaussian-mixture minority with exact samples. Compares threshold, perfect oversampling, SMOTE, and tilts by each utility and budget. Writes `runs/oracle/results_<tag>.csv` and `summary_<tag>.csv` (gain vs threshold, 95% CI). | `--n-min` (default 20 50 100) · `--seeds` (count, default 20) · `--learners lr xgb` · `--budgets` (default 0.1 0.25 0.5) · `--ir` (default 20) · `--dim` (default 4) · `--sep` (default 2.0) · `--unimodal` |

```bash
tilt oracle --n-min 20 50 100 --seeds 30 --learners lr xgb --budgets 0.1 0.25 0.5
```

```bash
tilt oracle --unimodal --seeds 30
```

```bash
tilt oracle --ir 50 --dim 8 --sep 1.5 --seeds 30
```

## 6. One method: train and infer

### Flags shared by `train`, `run` and `infer`

| flag | meaning |
|---|---|
| `--datasets` | table, comma list or group |
| `--seeds` | seed list, e.g. `--seeds 0 1 2` |
| `--config` | `configs/experiments/<name>.yaml`, or a path |
| `--model` | backbone key |
| `--hardware` | profile, or `auto` |
| `--finetune-mode auto\|full\|lora\|qlora` | fine-tuning mode |
| `--epochs` · `--lr` · `--temperature` | training and sampling settings |
| `--set k=v [k=v ...]` | any method config field |
| `--protocol clean\|paper` | whether a real validation fold is used |
| `--q` | kept share of the real minority |
| `--resume` | skip finished identical runs |
| `--shard i/N` | one process per GPU |
| `--strict` | stop at the first failure |
| `--dry-run` | list work items and exit |

Precedence: the method defaults, then the preset, then the `--config` YAML, then explicit flags, then `--set`.

**Configs:**

| config | machine | backbone | tables | seeds | q |
|---|---|---|---|---|---|
| `laptop_dev` | 4050 | qwen3-0.6b | natural_fast | 3 | 0.8 |
| `laptop_main` | 4050 | qwen3-1.7b | natural_fast | 5 | 0.8 |
| `h100_main` | H100 | qwen3-1.7b | natural | 10 | 0.8 |
| `h100_scale` | H100 | the ladder | natural_fast | 3 | 0.8 |
| `replication` | either | qwen3-1.7b | paper | 5 | 0.2 |
| `paper` | either | distilgpt2 (ImbLLM as published) | paper | 10 | 0.2 |
| `smoke` | – | tiny-random | synthetic | 1 | – |

### `tilt train`: one method on tables × seeds

It fits, generates and evaluates. Beyond the shared flags it takes:
- `--method` (required);
- `--preset`;
- `--experiment NAME`, which sets the run folder `runs/<NAME>/` (default `train`).

| `--method` | `--preset` values | what it is |
|---|---|---|
| `tilt` | `tilt_sir` (default) · `tilt_smc` · `tilt_off` · `tilt_influence` · `tilt_epistemic` · `tilt_aleatoric` · `tilt_random` · `tilt_contrast` · `tilt_nogate` | **the thesis method** (docs/METHOD.md) |
| `v2` | `recommended` (default) · `v1_exact` · `geometry_only` · `full` | ImbLLM, corrected (palt3) |
| `v1` | – | ImbLLM as published |
| `great_style` · `taptap_style` · `predllm_style` | – | LLM baselines (re-implementations) |
| `smote_nc` · `borderline_smote` · `svm_smote` · `smote` · `adasyn` · `ros` | – | classical oversamplers |
| `ctgan` · `tvae` · `tabddpm` · `arf` | – | deep generators (optional extras) |
| `threshold_only` · `prior_shift` · `scale_pos_weight` · `imbalance` · `original` | – | references: no generation |

**TILT `--set` fields:**

| field | values / default |
|---|---|
| `sampler` | `off` \| `sir` \| `smc` |
| `utility` | `fisher` \| `influence` \| `epistemic` \| `aleatoric` \| `none` \| `contrast` |
| `kl_budget` | 0.25 |
| `amp_cap` | 20 |
| `pool_factor` | 4 |
| `pilot` | 256 |
| `particles` | 8 |
| `keep` | 4 |
| `ess_frac` | 0.5 |
| `field_tokens` | 24 |
| `valid_q` | 0.05 |
| `privacy_alpha` | 0.05 |
| `probe_l2` | 0.01 |
| `contrast_sign` | ±1 |
| `majority_rows` | 0 |

Plus every V2Config field: `interpolation_mode`, `tau`, `snap_discrete`, `ordinal_interpolation`, `gibbs_sweeps`, `epochs`, `lr`, `temperature`, ...

```bash
tilt train --method tilt --preset tilt_sir --config laptop_dev
```

```bash
tilt train --method tilt --preset tilt_smc --datasets pc1 --seeds 0 --config laptop_dev
```

```bash
tilt train --method tilt --preset tilt_sir --config laptop_dev --set kl_budget=0.5 utility=epistemic
```

```bash
tilt train --method v2 --preset recommended --config laptop_dev --experiment compare
```

```bash
tilt train --method threshold_only --config laptop_dev --experiment compare
```

```bash
tilt train --method borderline_smote --datasets natural --seeds 0 1 2 --q 0.8
```

### `tilt infer`: fit (or load the cached fine-tune) and print generated rows next to real ones

It also prints the parse statistics and the plan / cache status. Flags:
- `--method` (an LLM method; default `v2`);
- `--preset`;
- `--dataset` (default `synthetic`);
- `--n` (rows to print, default 10);
- `--seed`;
- plus the shared flags.

```bash
tilt infer --method tilt --preset tilt_sir --dataset pc1 --n 10 --config laptop_dev
```

```bash
tilt infer --method v2 --dataset car_vgood --n 5 --config laptop_dev
```

## 7. Experiment grids

`tilt run EXPERIMENT` takes the shared flags from §6 and writes to `runs/<EXPERIMENT>/<table>/<method>/seed<k>/`. Each grid has an anchor, which is the method `tilt report` compares everything against.

### TILT (palt4, the thesis; docs/EXPERIMENTS.md)

| experiment | arms | anchor |
|---|---|---|
| `thesis_main` | original, imbalance, threshold_only, prior_shift, scale_pos_weight, smote_nc, borderline_smote, imbllm_paper, v2_recommended, v2_recommended+cllm, tilt_sir, tilt_smc | `threshold_only` |
| `utility` | tilt_fisher, tilt_influence, tilt_epistemic, tilt_aleatoric, tilt_none, tilt_contrast | `tilt_none` |
| `budget` | eps_0, 0.05, 0.1, 0.25, 0.5, 1, 2 | `eps_0` |
| `sampler` | sir_x2 / x4 / x8, smc_p4 / p8 / p16 | `sir_x4` |
| `validity` | tilt_sir, tilt_nogate, valid_0.2, no_dcr_gate | `tilt_sir` |
| `scale` | v2@ and tilt_sir@ each of qwen3-0.6b, 1.7b, 4b, 8b, 14b, qwen2.5-32b | – |
| `scarcity` | threshold_only, smote_nc, v2, tilt_sir at q = 0.05 / 0.1 / 0.2 / 0.5 | – |

### ImbLLM analysis (inherited from palt3)

| experiment | arms | anchor |
|---|---|---|
| `baselines` | references + classical (+ CTGAN / TVAE / TabDDPM / ARF if installed) | `imbalance` |
| `paper` | ImbLLM as published | – |
| `llm_baselines` | great_style, taptap_style, predllm_style | `great_style` |
| `ablation` | the 8 rows of ImbLLM Table III + Table IV | `full_imbllm` |
| `v2_compare` | v1_exact, recommended, full | `v1_exact` |
| `v2_flags` | each v2 flag added to v1 / removed from recommended | `v1_exact` |
| `interp` | 5 interpolation geometries | `interp_random` |
| `tau` | τ in {0, .25, .5, .75, 1} | `tau_1.0` |
| `temperature` | T in {0.5, 0.7, 1.0, 1.3} | `T_0.7` |
| `gibbs` | SIR-Gibbs sweeps 0 / 1 / 2 | `gibbs_0` |
| `prior2x2` | {pretrained, random init} × {named, anonymised columns} | `pretrained_named` |
| `order` | order sensitivity: permute_xy / fix_y / identity fine-tunes | `order_fix_y` |
| `curation` | {ImbLLM, v2, SMOTE-NC} × {raw, CLLM-curated} | `imbllm_paper` |
| `qsweep` | q in {.1 … .5} for imbalance / SMOTE-NC / ImbLLM | – |
| `rsweep` | interpolation ratio r in {0 … 1} | `r_1.0` |
| `smoke` | every code path, tiny backbone (`tilt smoke` runs this and checks artefacts) | – |

**The thesis sequence**, in the order of docs/EXPERIMENTS.md:

1. E1, the headline table:

   ```bash
   tilt run thesis_main --config h100_main --resume
   ```

2. E2, which utility:

   ```bash
   tilt run utility --config h100_main --resume
   ```

3. E3, the budget frontier:

   ```bash
   tilt run budget --config h100_main --resume
   ```

4. E4, SIR vs SMC:

   ```bash
   tilt run sampler --config h100_main --resume
   ```

5. E5, the validity defence:

   ```bash
   tilt run validity --config h100_main --resume
   ```

6. E6, backbone scale:

   ```bash
   tilt run scale --config h100_scale --resume
   ```

7. E7, scarcity:

   ```bash
   tilt run scarcity --config h100_main --resume
   ```

8. E8, the ImbLLM tables:

   ```bash
   tilt run thesis_main --config replication --resume
   ```

**Laptop rehearsal and planning:**
- List a grid without running it:

  ```bash
  tilt run thesis_main --config laptop_dev --dry-run
  ```

- A quick look on one table and one seed:

  ```bash
  tilt run thesis_main --config laptop_dev --datasets pc1 --seeds 0
  ```

- A sweep with XGBoost as the only learner (Git Bash / Linux syntax):

  ```bash
  TILT_LEARNERS=xgb tilt run budget --config laptop_dev --seeds 0
  ```

**Multi-GPU:** one process per GPU.

```bash
CUDA_VISIBLE_DEVICES=0 tilt run thesis_main --config h100_main --shard 0/2 --resume
```

```bash
CUDA_VISIBLE_DEVICES=1 tilt run thesis_main --config h100_main --shard 1/2 --resume
```

## 8. Reports

| command | what it prints / writes | flags |
|---|---|---|
| `tilt report EXPERIMENT` | datasets × methods (mean ± sd over seeds), run status, Friedman + Nemenyi critical difference, Holm-corrected Wilcoxon vs the anchor with bootstrap CI and the minimum attainable p, a gain-vs-covariate Spearman table, a cost table → `runs/<exp>/_report/` | `--metric` (default `f1`) · `--anchor` (override) · `--include-original` (rank `original` too) · `--figure` (also draw the report figure) |

**Metrics**, meaning any numeric key of `metrics.json`:

| kind | keys |
|---|---|
| headline | `gain_vs_threshold`, `real_gain_vs_threshold` |
| downstream | `f1`, `f1_tuned`, `auc`, `auprc`, `macro_f1`, `bacc` |
| per learner | `lr_*`, `rf_*`, `mlp_*`, `lgbm_*` |
| with the real minority rows | `real_*` |
| quality | `heldout_close_probability`, `heldout_coverage`, `dup_minor`, `nn_ratio` |
| privacy | `mia_auc`, `mia_auc_kde`, `mia_advantage` |
| TILT diagnostics | `tilt_lambda`, `tilt_kl`, `tilt_ess`, `tilt_valid_share`, `tilt_z_selected_mean` |
| order sensitivity | `order_std_per_token_synthetic`, `order_ratio_synthetic_to_real` |
| cost | `cost_elapsed_sec`, `cost_peak_vram_gb`, `cost_train_tokens`, `cost_gen_tokens` |
| covariates | `cov_*` |
| other | `shortfall` |

```bash
tilt report thesis_main --metric gain_vs_threshold
```

```bash
tilt report thesis_main --metric lr_f1_tuned --anchor threshold_only
```

```bash
tilt report thesis_main --metric f1 --include-original --figure
```

```bash
tilt report budget --metric heldout_close_probability
```

```bash
tilt report budget --metric mia_auc
```

```bash
tilt report sampler --metric cost_gen_tokens
```

```bash
tilt report utility --metric gain_vs_threshold --anchor tilt_none
```

## 9. Housekeeping

| command | what it deletes | flags |
|---|---|---|
| `tilt clean` | cached fine-tunes and/or experiment run folders | `--models` (everything in `runs/_models/`) · `--runs EXP [EXP ...]` |

```bash
tilt clean --models
```

```bash
tilt clean --runs smoke train
```

## 10. Make shortcuts

`make help` lists everything.

**Variables:**

| variable | default |
|---|---|
| `DATASETS` | natural_fast |
| `SEEDS` | 0 … 9 |
| `MODEL` | qwen3-1.7b |
| `HW` | auto |
| `CONFIG` | laptop_dev |
| `DS` | pc1 |
| `METRIC` | gain_vs_threshold |
| `EXP` | – |

| group | targets |
|---|---|
| install | `install` (core) · `install-llm` · `install-all` (llm + quant + sdv + kaggle) · `install-synthcity` · `install-flash` |
| data | `data-list` · `data-import` (from ../palt3) · `data-fetch` · `data-verify` · `data-inspect DS=` · `viz-data DATASETS=` |
| checks | `test` (fast) · `test-llm` · `theory` · `audit` · `oracle` · `audit-contamination MODEL=` · `smoke` · `check` (test + audit) |
| LLM switch | `models` · `plan MODEL= HW=` · `hw` · `pull MODEL=` · `where-llm` |
| TILT experiments | `run-main` · `run-utility` · `run-budget` · `run-sampler` · `run-validity` · `run-scale` · `run-scarcity` · `run-replication` (all take `CONFIG=`) |
| ImbLLM experiments | `run-baselines` · `run-paper` · `run-llm-baselines` · `run-ablation` · `run-v2` · `run-flags` · `run-interp` · `run-tau` · `run-temperature` · `run-gibbs` · `run-prior` · `run-qsweep` · `run-rsweep` · `run-order` · `run-curation` |
| reports | `report EXP= METRIC=` |
| housekeeping | `clean-models` · `help` |

```bash
make oracle
```

```bash
make run-main CONFIG=h100_main
```

```bash
make report EXP=thesis_main METRIC=gain_vs_threshold
```
