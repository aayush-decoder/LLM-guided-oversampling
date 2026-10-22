# Command reference

This file is self-contained: section 0 takes a fresh machine to a verified install, and section 4
confirms that both versions work end to end. Every later section assumes that:

* you are in the repo root (`D:\CODING\Amazon_ML\palt`, or wherever you cloned it), and
* the virtual environment is active (step 0.3).

v1 and v2 accept the same flags: `--dataset --datasets --method --methods --llm --mode --device --runs --seeds
--epochs --max-steps --binarize --max-rows --set a.b=v --plots --overwrite --run --n --out`.

---

## 0. Setup guide

### 0.1 Prerequisites

| need | version | check |
|---|---|---|
| Python | 3.10 – 3.12 (3.12 recommended) | `python --version` (Windows: `py -0p` lists installs) |
| NVIDIA driver (optional, for GPU) | any recent; CUDA ≥ 12.6 recommended | `nvidia-smi`, and read "CUDA Version" top right |
| uv (recommended) or pip | uv ≥ 0.4 | `uv --version`. Install with `pip install uv` if it is missing |
| Disk space | ~6 GB for PyTorch + deps; LLMs extra (distilgpt2 0.35 GB, 1.5B ≈ 3 GB, 3B ≈ 6 GB, 8B ≈ 16 GB) | |

No GPU? Everything runs on CPU. It is slow but fine for the smoke tests and the `tiny` / `distilgpt2` LLMs.

### 0.2 Go to the repo
```powershell
cd D:\CODING\Amazon_ML\palt
```

### 0.3 Create and activate the virtual environment

Windows (PowerShell):
```powershell
uv venv --python 3.12 .venv
.venv\Scripts\Activate.ps1
```
If PowerShell refuses to run `Activate.ps1` (execution policy), skip activation and prefix every
command in this file with the venv interpreter instead, e.g.
`.venv\Scripts\python.exe -m tabgen audit env`. uv commands can target it with
`uv pip install --python .venv\Scripts\python.exe ...`.

Linux / macOS:
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

Without uv, run `python -m venv .venv`, activate it the same way, and use `pip` wherever this file says `uv pip`.

### 0.4 Install PyTorch (pick ONE line that matches `nvidia-smi` → "CUDA Version")

| your driver shows | command |
|---|---|
| CUDA 12.8 or newer (RTX 40xx/50xx, recent drivers) | `uv pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| CUDA 12.6 – 12.7 | `uv pip install torch --index-url https://download.pytorch.org/whl/cu126` |
| no NVIDIA GPU | `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` |

For older drivers, pick the matching wheel on https://pytorch.org/get-started/locally/. The CUDA
download is about 2.6 GB.

### 0.5 Install the project and its dependencies
```bash
uv pip install -r requirements.txt
uv pip install -e .
```
* `-e .` installs the three packages `tabgen`, `predllm_v1` and `predllm_v2` in editable mode, so
  edits in `common/`, `v1/` and `v2/` apply immediately.
* `bitsandbytes` is only used for QLoRA (the 3B–8B models). If it fails to install on your platform,
  everything else still works; use `--mode lora` or smaller models.
* `sdv` is only used for the CTGAN / TVAE / CopulaGAN baselines. Without it, those methods are
  skipped in smoke tests and raise a clear error in benchmarks.

### 0.6 Optional configuration

| what | when | how |
|---|---|---|
| Where LLM weights are cached | the system drive is small | set `HF_HOME` before running, e.g. PowerShell `$env:HF_HOME="D:\hf_cache"` / bash `export HF_HOME=/data/hf_cache` |
| Gated LLMs (`llama3.2-1b`, `llama3.2-3b`, `llama3.1-8b`) | you want Llama | accept the licence on the model's Hugging Face page, then `huggingface-cli login` |
| Kaggle datasets (`gem_price`, `credit_card`) | you benchmark the extended set | `uv pip install kaggle` and put `kaggle.json` in `%USERPROFILE%\.kaggle\` (or `~/.kaggle/`), or drop the CSVs manually (see `docs/DATASETS.md`) |
| Offline re-runs | the models are already downloaded | set `HF_HUB_OFFLINE=1` |

Nothing is downloaded at install time. Datasets download on first use (`data/raw/`) and LLMs on
first use (Hugging Face cache).

### 0.7 Verify the install (no training, under a minute)
```bash
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
python -m tabgen audit env               # PASS/WARN per package, GPU name + memory, LLMs that fit
python -m tabgen llm list                # LLM switch table; "fits" column for your GPU
python -m pytest -m "not smoke"          # offline unit tests (iris ships with scikit-learn)
python -m tabgen llm check --llm tiny    # downloads a few-MB model: 1 train step + 1 generation
```
Expected: `cuda: True` on a GPU machine, `audit env` ends without `FAIL`, and every pytest test passes.
On a GPU, also run `python -m tabgen llm check --llm distilgpt2`, plus `--llm 1.5b` if you plan to use it.

### 0.8 Setup troubleshooting

| symptom | fix |
|---|---|
| `cuda: False` although you have an NVIDIA GPU | you installed the CPU wheel; rerun step 0.4 with the matching `cu12x` index (add `--reinstall`) |
| `bitsandbytes` import error | only QLoRA needs it; use `--mode lora` or models ≤ 1.7B |
| `401` / `gated repo` when loading Llama | step 0.6: accept the licence + `huggingface-cli login` |
| `ModuleNotFoundError: tabgen` | the venv is not active, or `uv pip install -e .` was skipped |
| CUDA out of memory | smaller `--llm`, or `--set finetune.micro_batch_size=2`, or `--mode qlora` |
| slow / blocked Hugging Face downloads | set `HF_HOME` to a fast disk; behind a proxy, set `HTTPS_PROXY` |

## 1. Data
```bash
python -m tabgen data download --dataset all    # auto; see docs/DATASETS.md for the 2 Kaggle-only sets
python -m tabgen data list
python -m tabgen data info --dataset adult      # sizes, classes, cleaning report
```

## 2. Visualise data before/after cleaning, preprocessing and feature encoding
```bash
python -m tabgen viz data --dataset adult                      # raw + clean + encoded -> reports/data_viz/adult/
python -m tabgen viz data --dataset adult --stages raw         # only the raw (pre-cleaning) plots
python -m tabgen viz data --dataset paper                      # every paper dataset
```
Outputs: `raw_missing.png`, `raw_distributions.png`, `clean_distributions.png`,
`encoded_distributions.png`, `encoded_correlation.png`, and `serialized_examples.txt` (the exact text
the v1 and v2 LLMs are trained on).

## 3. LLM switch
```bash
python -m tabgen llm list
python -m tabgen llm check --llm 1.5b           # 1 train step + 1 generation, peak memory
```

## 4. Smoke tests (fast end-to-end run of each version, every artifact verified)

Run these right after section 0. They go through the whole pipeline (split → fine-tune → generate →
TSTR + fidelity metrics → plots → run audit → result table) on **iris**, which ships with
scikit-learn, so no dataset download is needed. Each run uses one seed and a short fine-tune.
**The scores are meaningless.** A smoke test checks that the code works, not how good it is.

```bash
python -m predllm_v1 smoke
python -m predllm_v2 smoke
```

| | v1 smoke | v2 smoke |
|---|---|---|
| LLM (first run downloads it) | `distilgpt2` (~350 MB) | `tiny-gpt2` (a few MB, random weights) |
| fine-tuning | 150 steps | 30 steps |
| methods | original, pred_llm, great, taptap, ctgan (skipped if `sdv` is missing) | original, pred_llm_v2, v2_encoded, v2_plain_loss |
| rough time | 1–3 min on GPU, 5–15 min on CPU | about 1 min on GPU, a few min on CPU |
| why this LLM | v1 decodes free text, so the LLM must first learn the "X is v" format; a random model produces no valid rows | v2's decoder cannot produce invalid rows, so even a random model completes the pipeline |

**What counts as a pass** (checked in code, `common/tabgen/smoke.py`). For every method:
* the run finished without an exception;
* these files exist in its `seed0/` directory: `config.json`, `split.json`, `preprocessor.json`,
  `synthetic_encoded.csv`, `synthetic_clean.csv`, `generation_stats.json`, `metrics.json`,
  `plots/marginals.png`, `plots/tsne.png` and `plots/dcr.png`; LLM methods also need
  `model_ref.json` and `train_log.jsonl`;
* the run audit (`audit.json`) has no `FAIL`: no NaNs, every class generated, and a valid TSTR score.

**Output.** The console prints one `PASS`/`FAIL` line per method and `OVERALL: PASS|FAIL`. The exit
code is `0` only when everything passes. Files are written to:
```
v1/runs/_smoke/SMOKE_REPORT.md                    # the same summary
v1/runs/_smoke/iris/<method>__<llm>/seed0/...     # full run artifacts
v1/runs/_smoke/iris/_models/...                   # fine-tuned smoke models
v1/runs/_smoke/tables/table_accuracy.md           # results-table code path
(same layout under v2/runs/_smoke/)
```
Smoke runs live in `runs/_smoke/`, so they never mix with, or get resumed as, real results.
Re-running overwrites them. Delete `v1/runs/_smoke` and `v2/runs/_smoke` to clean up.

**Variants**
```bash
python -m predllm_v1 smoke --device cpu                        # force CPU
python -m predllm_v1 smoke --max-steps 300                     # more steps if validity is too low on your machine
python -m predllm_v2 smoke --llm distilgpt2 --max-steps 150    # v2 with a real (pretrained) LLM
python -m predllm_v2 smoke --llm tiny-llama                    # exercises the LoRA path on a Llama-style model
python -m pytest -m smoke                                      # same checks, run as pytest tests
```

**Smoke troubleshooting**

| symptom in SMOKE_REPORT.md | meaning / fix |
|---|---|
| `SKIP ctgan: sdv not installed` | fine; install `sdv` to cover the baselines too |
| v1 `RuntimeError: LLM produced no valid rows` | the fine-tune was too short for your setup; use `--max-steps 300` |
| audit `FAIL: classes generated [...]` | same cause (under-trained v1 model); more steps |
| `missing=[...plots/tsne.png]` | a plotting error; the traceback is printed above the report |
| model download errors | see 0.8 (proxy / `HF_HOME` / connectivity) |

## 5. Tests
```bash
python -m pytest -m "not smoke"                 # offline unit tests (seconds)
python -m pytest -m smoke                       # end-to-end smoke tests (downloads small LLMs)
python -m pytest                                # everything
```

## 6. Training + generation + evaluation (v1 = paper replication)
```bash
python -m predllm_v1 methods                                        # presets = Table III rows
python -m predllm_v1 run --dataset iris --method pred_llm           # 3 seeds, distilgpt2, 50 epochs (paper)
python -m predllm_v1 run --dataset adult --methods pred_llm,great,taptap --plots
python -m predllm_v1 run --dataset adult --llm 1.5b --epochs 5      # bigger LLM (LoRA)
python -m predllm_v1 run --dataset housing --binarize median        # K=2 variant of a regression set
python -m predllm_v1 benchmark --datasets paper --methods paper     # Table II (Original, SDV GANs, GReaT, TapTap, Pred-LLM)
python -m predllm_v1 benchmark --datasets user --methods paper      # the extended 10 datasets
python -m predllm_v1 ablation --datasets classification             # Table III
python -m predllm_v1 train --dataset adult --method pred_llm --seeds 0   # fine-tune only
```

## 7. Training (v2 = improved method)
```bash
python -m predllm_v2 methods
python -m predllm_v2 run --dataset adult --method pred_llm_v2 --plots
python -m predllm_v2 benchmark --datasets paper,user --methods compare   # Original, v2, CTGAN, TVAE
python -m predllm_v2 ablation --datasets classification                 # N1..N5 switched off one at a time
python -m predllm_v2 run --dataset adult --llm 3b --set query.permutations=2
```

## 8. Inference testing (on a finished seed dir)
```bash
python -m predllm_v1 generate --run v1/runs/adult/pred_llm__distilgpt2/seed0 --n 5000
python -m predllm_v1 infer    --run v1/runs/adult/pred_llm__distilgpt2/seed0    # LLM as classifier on real test
python -m predllm_v2 generate --run v2/runs/adult/pred_llm_v2__distilgpt2/seed0 --n 5000
python -m predllm_v2 infer    --run v2/runs/adult/pred_llm_v2__distilgpt2/seed0
python -m predllm_v1 evaluate --run <seed dir>                                  # recompute metrics
```

## 9. Visualise results
```bash
python -m tabgen viz run --run v1/runs/adult/pred_llm__distilgpt2/seed0   # marginals, t-SNE, DCR, Fig. 2 heatmaps
python -m tabgen viz run --run v2/runs                                     # every finished run
python -m predllm_v1 attention --run v1/runs/adult/pred_llm__distilgpt2/seed0   # Fig. 4
```

## 10. Audits
```bash
python -m tabgen audit env
python -m tabgen audit data --dataset all       # sizes vs expected, missing, duplicates, imbalance, overlap, leakage
python -m tabgen audit run --run v1/runs        # artifacts, validity, class coverage, memorisation, label shift
python -m tabgen audit run --run v2/runs
```
Reports are written to `reports/audits/*.json` and `<run>/audit.json`.

## 11. Result tables (Table II / III / Fig. 5)
```bash
python -m tabgen report --versions v1,v2        # -> reports/tables/table_accuracy.md, table_mse.md, fidelity.md
```
