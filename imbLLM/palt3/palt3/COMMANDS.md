# palt3 commands

Every command is `imb <command>` (or `python -m imbllm <command>`), run from the `palt3/` directory. Add `-h` to any command for its flags. Every `make` target in the Makefile wraps one of these.

Nothing downloads until you run a command that needs it: `imb data fetch` for data, and `imb pull` or the first LLM run for model weights.

---

## 0. Setup

| what | command |
|---|---|
| core (data, audits, classical baselines, evaluation) | `pip install -e ".[dev]"` |
| + any LLM method | `pip install -e ".[llm,dev]"` |
| + 4-bit QLoRA, CTGAN/TVAE, Kaggle | `pip install -e ".[llm,quant,sdv,kaggle,dev]"` |
| + TabDDPM / ARF baselines, LightGBM learner | `pip install -e ".[synthcity,lgbm]"` (heavy; use a separate env if synthcity pins a different torch) |
| + FlashAttention-2 (A100/H100) | `pip install -e ".[flash]" --no-build-isolation` |

Environment variables:

| variable | effect |
|---|---|
| `IMBLLM_DATA`, `IMBLLM_RUNS`, `IMBLLM_CACHE` | move data, runs or the model cache |
| `IMBLLM_LEARNERS` | downstream learners, e.g. `xgb` (fast) or `xgb,lr,rf,mlp,lgbm`. Default: `xgb,lr,rf,mlp` (+ `lgbm` if installed) |
| `HF_HOME`, `HF_TOKEN` | Hugging Face cache and gated-model access (Llama) |

## 1. Data (before/after cleaning, protocol)

```bash
imb data list
```

```bash
imb data fetch --datasets all --from ../palt/data/raw
```

```bash
imb data fetch --datasets open
```

```bash
imb data verify --datasets all
```

```bash
imb data inspect --dataset adult --export
```

```bash
imb viz data --datasets fast
```

`viz data` writes raw-vs-clean, protocol-funnel, class-balance and minority-geometry figures to `data/figures/<dataset>/`.

## 2. LLM switch

```bash
imb models
```

```bash
imb plan --model qwen2.5-3b --hardware rtx4050_6gb
```

```bash
imb hw
```

```bash
imb pull --model distilgpt2 qwen2.5-1.5b
```

```bash
imb where-llm --grep
```

## 3. Tests, audits, smoke

```bash
pytest -q -m "not llm and not slow"
```

```bash
pytest -q -m "not slow"
```

```bash
imb audit all
```

```bash
imb audit contamination --model distilgpt2 --datasets adult,german,bank,compas
```

```bash
imb smoke
```

- The first pytest command needs no torch, network or data.
- The second pytest command adds the end-to-end LLM tests on the offline tiny backbone.
- `imb audit all` runs the math, paper-table and data checks; it needs no LLM.
- `imb audit contamination` loads the pre-trained backbone and runs greedy memorisation probes. Results go to `runs/_audits/contamination_<model>.json`.

## 4. One method: train, then infer

```bash
imb train --method v2 --preset recommended --datasets fast --seeds 0
```

```bash
imb train --method tabddpm --datasets fast --seeds 0
```

```bash
imb infer --method v1 --dataset australian --n 10
```

`--method` accepts:

| type | values |
|---|---|
| LLM | `v1`, `v2` (with `--preset`), `great_style`, `taptap_style`, `predllm_style` |
| classical | `ros`, `smote`, `adasyn`, `smote_nc` |
| deep | `ctgan`, `tvae`, `tabddpm`, `arf` |
| references | `original`, `imbalance`, `threshold_only`, `prior_shift`, `scale_pos_weight` |

## 5. Experiments

Every grid takes `--datasets --seeds --config --model --hardware --set k=v --protocol --q --resume --shard i/N --dry-run`. Configs in `configs/experiments/`: `paper` (distilgpt2, 10 seeds), `fast`, `smoke`, `laptop_qwen` (Qwen2.5-1.5B), `h100_scale` (Qwen3-8B).

| experiment | question it answers | command |
|---|---|---|
| `baselines` | non-LLM columns of Table II plus the threshold / prior-shift / class-weight references | `imb run baselines --config paper` |
| `paper` | ImbLLM as published | `imb run paper --config paper` |
| `llm_baselines` | GReaT- / TapTap- / Pred-LLM-style (re-implementations) | `imb run llm_baselines --config paper` |
| `ablation` | Table III + IV with seed variance | `imb run ablation --config paper` |
| `v2_compare` | v1_exact vs recommended vs full | `imb run v2_compare --config paper` |
| `v2_flags` | per-flag attribution | `imb run v2_flags --config paper` |
| `interp`, `tau`, `temperature`, `gibbs` | one v2 knob each | `imb run tau --config paper` |
| `order` | is the fine-tuned LM order-inconsistent, relative to real rows and to a fixed-order model? | `imb run order --config paper` |
| `curation` | does CLLM curation help ImbLLM more than SMOTE-NC? | `imb run curation --config paper` |
| `prior2x2` | pretrained vs random init × named vs anonymised columns | `imb run prior2x2 --config paper` |
| `qsweep`, `rsweep` | Fig. 6 / Fig. 5 | `imb run qsweep --config paper` |

Multi-GPU: start one process per GPU with `--shard 0/4`, `--shard 1/4`, and so on.

## 6. Reports

```bash
imb report paper --metric f1
```

```bash
imb report baselines --metric lr_f1
```

```bash
imb report order --metric order_ratio_synthetic_to_real
```

```bash
imb report v2_flags --metric real_f1 --figure
```

- **Metrics:** any numeric key of `metrics.json`.
  - Paper-comparable: `f1`, `auc`.
  - Prefixed variants: `real_*` (with the real minority rows), `<learner>_*` (e.g. `lr_f1`, `rf_auprc`, `mlp_f1_tuned`).
  - Other scores: `auprc`, `f1_tuned`, `macro_f1`, `bacc`, `heldout_close_probability`.
  - Order sensitivity: `order_std_per_token_synthetic`, `order_ratio_synthetic_to_real`.
  - Cost: `cost_elapsed_sec`, `cost_peak_vram_gb`.
- **Every report also prints:**
  - Friedman/Nemenyi;
  - Holm-corrected paired tests against the anchor;
  - a Spearman table of gain vs dataset covariates (`cov_*`, when there are at least 5 datasets);
  - a cost table.

## 7. Housekeeping

```bash
imb clean --models
```

```bash
imb clean --runs smoke
```
