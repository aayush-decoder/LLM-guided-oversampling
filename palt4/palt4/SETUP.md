# palt4 setup

This takes you from a clean machine to a passing smoke test, on the **RTX 4050 laptop (Windows 11)** and on the **H100 server (Linux)**. Every command you then run is in [COMMANDS.md](COMMANDS.md).

Nothing here trains a model. The first real training run is step 8.

---

## 0. What you need

| item | laptop (RTX 4050 6 GB, Windows 11) | H100 80 GB server (Linux) |
|---|---|---|
| Python | 3.10–3.12 (3.12 recommended), from python.org or `uv` | 3.10–3.12 |
| NVIDIA driver | any recent driver; `nvidia-smi` must work | same |
| Git | optional (the repo is a folder) | optional |
| Disk: code + environment | ~6 GB (torch with CUDA is large) | ~6 GB |
| Disk: model cache (`HF_HOME`) | Qwen3-0.6B ~1.2 GB · 1.7B ~3.5 GB · 4B ~8 GB | 8B ~16 GB · 14B ~30 GB · Qwen2.5-32B ~65 GB |
| Disk: data + runs | ~1 GB tables; runs grow with experiments (keep 20 GB free) | put `TILT_RUNS` on scratch |
| RAM | 16 GB is enough | – |
| Network | only while installing, fetching data and pulling models | same |

**Which backbone runs where.** `tilt models` prints this after install; the details are in `docs/COMPUTE.md`.

| machine | backbones |
|---|---|
| laptop | Qwen3-0.6B (development) and Qwen3-1.7B (thesis backbone, 4-bit QLoRA) |
| H100 | everything up to Qwen3-14B in bf16 LoRA, plus Qwen2.5-32B with QLoRA |

---

## 1. Open the project

The project lives at `D:\CODING\Amazon_ML\palt4` on the laptop. Copy the whole `palt4/` folder to the server (for example with `scp -r` or `rsync`).

**Laptop (PowerShell):**

```powershell
cd D:\CODING\Amazon_ML\palt4
```

**Server:**

```bash
cd ~/palt4
```

---

## 2. Create a virtual environment

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script, allow it for your user once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Git Bash alternative:**

```bash
source .venv/Scripts/activate
```

**Unicode on Windows.** Outputs contain τ, λ, ≈ and ε. Set UTF-8 once per user, then open a new terminal:

```powershell
[Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "User")
```

### Linux (H100)

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

### Alternative: `uv` (fast, both systems)

```bash
uv venv -p 3.12 .venv
```

With `uv`, replace `pip install` with `uv pip install` in the steps below.

---

## 3. Install PyTorch with CUDA

Install torch **before** palt4, so pip does not pull a CPU-only build.

1. Check the CUDA version your driver supports (top-right of the `nvidia-smi` output):

   ```bash
   nvidia-smi
   ```

2. Pick the wheel index for that version:

   | driver reports CUDA | install command |
   |---|---|
   | ≥ 12.8 (including 13.x) | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
   | 12.6–12.7 | `pip install torch --index-url https://download.pytorch.org/whl/cu126` |
   | no NVIDIA GPU | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

3. For the laptop's driver (CUDA 13.0):

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```

4. Check that the GPU is visible:

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   It must print `True` and the GPU name. If it prints `False`, see Troubleshooting (§10).

---

## 4. Install palt4

| machine | command | what it adds |
|---|---|---|
| laptop and H100 (thesis runs) | `pip install -e ".[llm,quant,dev]"` | transformers, peft, accelerate, bitsandbytes (QLoRA), pytest |
| CPU-only checks (theory, oracle, fast tests) | `pip install -e ".[dev]"` | core only: numpy, pandas, scikit-learn, xgboost, imbalanced-learn |
| + CTGAN / TVAE baselines | `pip install -e ".[sdv]"` | SDV |
| + LightGBM as a fourth-and-a-half learner | `pip install -e ".[lgbm]"` | lightgbm |
| + Kaggle-only tables (credit_card, gem_price) | `pip install -e ".[kaggle]"` | kaggle CLI |
| H100 only, optional: FlashAttention-2 | `pip install -e ".[flash]" --no-build-isolation` | needs a CUDA toolchain; SDPA is used otherwise |
| TabDDPM / ARF baselines | `pip install -e ".[synthcity,lgbm]"` | **in a separate venv**; synthcity pins older torch versions |

The thesis install:

```bash
pip install -e ".[llm,quant,dev]"
```

This installs the `tilt` command. If `tilt` is not found, `python -m tiltgen` does the same thing. Check the CLI works:

```bash
tilt -h
```

**Windows and bitsandbytes.** Official x86-64 CUDA wheels exist, so plain `pip` works. If QLoRA still fails, run the laptop with `--finetune-mode lora` and Qwen3-0.6B.

---

## 5. Model cache and Hugging Face

Model weights download on first use into `HF_HOME`.
- Move the cache to a big disk before the first pull. On the server, use scratch space.
- The Qwen checkpoints are **ungated**, so no token is needed.
- Llama checkpoints need `huggingface-cli login` and acceptance of the licence on the model page.

**Laptop (PowerShell, persistent; open a new terminal afterwards):**

```powershell
[Environment]::SetEnvironmentVariable("HF_HOME", "D:\hf_cache", "User")
```

**Server (add to `~/.bashrc`):**

```bash
export HF_HOME=/scratch/$USER/hf_cache
```

**Optional:** pre-download the backbones, so a timed H100 session never waits on a download.
- Laptop:

  ```bash
  tilt pull --model qwen3-0.6b qwen3-1.7b
  ```

- H100:

  ```bash
  tilt pull --model qwen3-1.7b qwen3-8b qwen3-14b qwen2.5-32b
  ```

---

## 6. Where results go

Defaults live inside the repo: `data/`, `runs/` and `runs/_models/` (cached fine-tunes). All three are git-ignored. Override them with environment variables when the disk is small, or on the server:

| variable | what it moves | example (server) |
|---|---|---|
| `TILT_DATA` | datasets | `/scratch/$USER/tilt/data` |
| `TILT_RUNS` | run folders and reports | `/scratch/$USER/tilt/runs` |
| `TILT_CACHE` | cached fine-tuned models | `/scratch/$USER/tilt/models` |
| `TILT_LEARNERS` | downstream learners | `xgb` for quick sweeps; default `xgb,lr,rf,mlp` |

---

## 7. Get the data

Tables download only when you run these.

1. **Thesis benchmark:** the 10 naturally imbalanced tables, fetched from OpenML (a few MB):

   ```bash
   tilt data fetch --datasets natural
   ```

2. **Optional:** reuse the tables palt3 already downloaded (offline). This is laptop only:

   ```bash
   tilt data fetch --datasets all --from ../palt3/data/raw
   ```

3. **Optional:** everything else reachable without credentials:

   ```bash
   tilt data fetch --datasets open
   ```

4. Check shapes against the registry. An N mismatch after dropping missing values is logged, not an error:

   ```bash
   tilt data verify --datasets natural
   ```

**Kaggle-only tables (credit_card, gem_price).** Put `kaggle.json` in `~/.kaggle/` (on Windows, `C:\Users\<you>\.kaggle\`), then fetch them:

```bash
tilt data fetch --datasets credit_card,gem_price
```

---

## 8. Check the installation

Run these in order. Each line says what a pass looks like.

| # | command | pass looks like | needs |
|---|---|---|---|
| 1 | `tilt hw` | your GPU, with its VRAM and bf16=True | – |
| 2 | `tilt models` | a table of backbones × hardware; Qwen3.5 / MoE listed as rejected | – |
| 3 | `pytest -q -m "not llm and not slow"` | `104 passed`, 0 failed | – |
| 4 | `tilt audit theory` | 6 × PASS, `ALL PASS` | – |
| 5 | `tilt audit all` | `ALL PASS` (data checks need step 7) | – |
| 6 | `pytest -q -m "not slow"` | the `llm` tests also pass (tiny offline model, a few minutes) | torch |
| 7 | `tilt smoke` | `SMOKE PASS`, including `tilt_sir` and `tilt_smc` | torch |
| 8 | `tilt plan --model qwen3-1.7b --hardware rtx4050_6gb` | a plan that fits, with mode and micro-batch | torch |

The **first real run**: three small tables, Qwen3-0.6B, tens of minutes on the laptop.

```bash
tilt train --method tilt --preset tilt_sir --config laptop_dev --seeds 0
```

Then check the result:

```bash
tilt report train --metric gain_vs_threshold
```

---

## 9. Working on the H100 (sessions of a few hours)

1. **Log in and open a persistent session**, so a dropped connection doesn't kill the run:

   ```bash
   tmux new -s tilt
   ```

   To reattach later:

   ```bash
   tmux attach -t tilt
   ```

2. **Activate the environment and set the scratch paths** (once per session):

   ```bash
   source .venv/bin/activate
   ```

   ```bash
   export TILT_RUNS=/scratch/$USER/tilt/runs TILT_CACHE=/scratch/$USER/tilt/models
   ```

3. **Run with `--resume`.** An interrupted experiment continues where it stopped, because finished runs are skipped by their config hash:

   ```bash
   tilt run thesis_main --config h100_main --resume
   ```

4. **Several GPUs:** one process per GPU, each taking a shard:

   ```bash
   CUDA_VISIBLE_DEVICES=0 tilt run thesis_main --config h100_main --shard 0/2 --resume
   ```

   ```bash
   CUDA_VISIBLE_DEVICES=1 tilt run thesis_main --config h100_main --shard 1/2 --resume
   ```

5. **Know the size before starting:** `--dry-run` lists the work items without running anything:

   ```bash
   tilt run thesis_main --config h100_main --dry-run
   ```

6. **Bring results home** to the laptop and report there. The reports need no GPU:

   ```bash
   rsync -av server:/scratch/$USER/tilt/runs/ ./runs/
   ```

   ```bash
   tilt report thesis_main --metric gain_vs_threshold
   ```

A suggested weekly split of the ~20 H100 hours is in `docs/COMPUTE.md`. Sampling-only experiments (`utility`, `budget`, `validity`) reuse cached fine-tunes, so they are cheap. Run them after `thesis_main`.

---

## 10. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `torch.cuda.is_available()` is False | CPU wheel installed, or the driver is too old for the wheel | `pip uninstall torch`, then reinstall from the index matching `nvidia-smi` (§3) |
| `KeyError: 'qwen3'` | transformers < 4.51 | `pip install -U "transformers>=4.51"` |
| bitsandbytes import / CUDA error | wheel does not match the GPU / CUDA | `pip install -U bitsandbytes`; or run the laptop with `--finetune-mode lora --model qwen3-0.6b` |
| CUDA out of memory | backbone too big for the device | check the plan with `tilt plan --model <key> --hardware <profile>`, then use a smaller rung, `--finetune-mode qlora`, or `--set effective_batch=16` (accumulation keeps training stable) |
| `tilt` refuses a model "with a reason" | the planner says it cannot fit, or the model is incompatible (Qwen3.5, MoE, instruct) | pick a listed Base model (`tilt models`) |
| `UnicodeEncodeError: 'charmap'` on Windows | console is not UTF-8 | set `PYTHONIOENCODING=utf-8` (§2) |
| `tilt: command not found` | editable install missing from this venv | activate the venv, then `pip install -e .`, or use `python -m tiltgen` |
| OpenML fetch fails or times out | network / OpenML down | retry; the cache lives under scikit-learn's data home; or copy `data/raw/` from another machine |
| `tilt smoke` shows low parse yield | the tiny model is under-trained on this machine | raise `epochs` in `configs/experiments/smoke.yaml` (for example to 100) |
| a run folder has `status: error` | see `manifest.json → error` in that run folder | fix, then rerun with `--resume` (only failed / missing items rerun) |
| reports show NaN `gain_vs_threshold` | no `threshold_only` twin for that (dataset, seed), or protocol `paper` | run `thesis_main` (it includes `threshold_only`) with protocol `clean` |
| H100 session ended mid-run | – | reattach tmux, or rerun the same command with `--resume` |

---

## 11. Clean up

| what | command |
|---|---|
| cached fine-tunes (can be large) | `tilt clean --models` |
| one experiment's run folders | `tilt clean --runs smoke` |
| downloaded backbones | delete the folder `HF_HOME` points to |
| everything | delete `.venv/`, `data/`, `runs/` |
