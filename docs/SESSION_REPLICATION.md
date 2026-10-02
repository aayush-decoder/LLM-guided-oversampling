# Session replication record

This file records how this repo was built: what was asked, what was decided, what was checked,
and what state was left behind. The repo is the output. Use this file to replay the session or to
continue it without guessing.

Built with Claude Code (desktop, Code tab), Opus 5.5, ending 2026-09-27.

---

## 1. Environment

| item | value |
|---|---|
| OS | Windows 11 Home (10.0.26200) |
| GPU | RTX 4050 Laptop, 6 GB VRAM, driver CUDA 13.0 |
| Python | 3.12, called as `py -3.12` because `python` is not on PATH |
| Env tool | `uv` (a `.venv` exists but is only partly built: torch is **not** installed) |
| Repo root | `D:\CODING\Amazon_ML\palt\` |
| Paper | `Generating Realistic Tabular Data with Large.pdf` in the repo root (arXiv:2410.21717) |
| PDF text | extracted with PyMuPDF (`fitz`) because `pdftoppm` is missing; the extract is in the session scratchpad as `paper.txt` |

---

## 2. User prompts, in order (verbatim apart from the pasted table)

1. **Build request**
   > @PDF replicate this paper, additionally add [dataset table] These datasets. [pasted] coding part. Have a well diverse and modular file structure, but dont make unnecessary long, just what required, do the coding part to replicate the paper first in v1, then in v2 we will add the novelity that we are making the better and improving the current paper and current market benchmarks, full aware with the significant architectural and mathematics changes in this v2 version, also mention them in the v2 docs/ . Make sure we have guides to download the datasets if it cant be automatically downloaded with a script. Also, make Smoke tests for both the versions and the required artifacts via the code. And the training produce the required in runs/ dir of respective versions. After all, it should be clear what each module of each versions is doing, also have a list of commands to start traning, inference testing, testing, smoke testing, running audits, or visualizing any data after/before data cleaning/pre-processing/feature-engineering and where the LLM is used. Please have modular LLM switch so we can switch from low to high parameter LLM (1.5B, 3B, 8B, etc) so we can run on various devices to test, if fine tuning is involved make sure to list the compatible LLMs only, and used them only. Ask any doubts and make the monorepo for both the versions in the codebase.

   Pasted dataset table (N, M, #num, #cat, K, source):

   | dataset | N | M | num | cat | K | source |
   |---|---|---|---|---|---|---|
   | adult | 30,162 | 11 | 5 | 6 | 2 | UCI |
   | german | 1,000 | 20 | 6 | 14 | 2 | UCI |
   | fuel | 639 | 8 | 3 | 5 | 2 | Kaggle |
   | insurance | 1,338 | 6 | 3 | 3 | 2 | Kaggle |
   | gem_price | 26,270 | 9 | 6 | 3 | 2 | Kaggle |
   | housing | 20,433 | 9 | 8 | 1 | 2 | Kaggle |
   | bank | 4,521 | 14 | 5 | 9 | 2 | [22] |
   | car | 1,728 | 6 | 2 | 6 | 4 | UCI |
   | credit_card | 10,127 | 19 | 14 | 5 | 2 | UCI |
   | sick | 3,560 | 27 | 6 | 21 | 2 | OpenML |

2. **Answers to the clarifying questions**

   | question | answer |
   |---|---|
   | Which datasets? | Paper's 20 + your 10 |
   | Regression datasets in the user's table (K = 2)? | Config flag, regression default (`--binarize median` is optional) |
   | Baselines? | LLM variants + SDV GANs (CTGAN, TVAE, CopulaGAN) |
   | v2 novelty? | You design it |

3. **Standing constraint (sent mid-build)**
   > no need to literally download the datasets or run the actual files or codes just do basic check, syntax check, and if its executable/compilable code, and we are good to go. Just it should download the stuff when we run the required command, hence the commands should be extensive, rather doing hard run everything at the spot. continue now.

   From then on, verification was limited to compile, YAML, lint and import checks. The two
   background jobs still running (dataset download, torch install) were stopped.

4. **Setup guide request**
   > add the setup guide too in COMMANDS.md along with smoke test , make sure it is self contained

5. **Feasibility question**
   > check if it is feasible for french vs english data (test) where french is the minority class and model have only 5% examples of french texts, how well it will perform, give a report.

   Answered statically in `docs/FEASIBILITY_FR_EN.md`.

6. `/compact`, then a request for an in-depth summary to re-create the session. The first attempt ran the wrong skill (`/sales:call-summary`); this file is the result.

---

## 3. Key decisions (and why)

| decision | reason |
|---|---|
| One shared package `common/tabgen` plus thin `v1/predllm_v1` and `v2/predllm_v2` | Both versions share the data, evaluation, baseline, plot and audit code; each version only adds its generator |
| Paper protocol copied from the authors' code: categories → codes, MinMax on every feature, 2 decimals, scaler fitted on the **full** data, stratified 80/20 split with seed = run (0, 1, 2) | Makes Table II/III comparable. `fit_scaler_on` is a config key |
| Paper rule: TSTR accuracy = 0 if D_fake is missing a class | Stated in the paper; implemented in `eval/tstr.py` |
| TapTap = GReaT + a classifier that labels the rows, as in Table III | Matches the paper's ablation |
| All GReaT, TapTap and ablation variants are presets of one v1 pipeline (`permutation` × `sampling` × `query` × `augment_original`) | One code path, and the ablation table is just a list of presets |
| Fine-tuned model cache `runs/<ds>/_models/<llmtag>_<sha12>`, keyed on a fingerprint of everything that affects fine-tuning | Presets that fine-tune identically reuse one model |
| LLM registry lists **only byte-level BPE** base models (GPT-2 family, Qwen2.5/3, Llama 3.x, SmolLM2) | The `X is v` scaffold and v2's numeric logit mask need tokens without a SentencePiece dummy-prefix space. Mistral was dropped for that reason |
| Modes: full (≤ 360M), LoRA (0.5B–1.7B), QLoRA NF4 (3B–8B); gradient accumulation to an effective batch of 32 | Fits the 6 GB GPU and scales to larger GPUs |
| Smoke runs use `runs/_smoke/` via `runs_subdir` | Keeps smoke output out of real results |
| v1 smoke uses distilgpt2 with 150 steps; v2 smoke uses tiny-gpt2 with 20–30 steps | A random tiny model produces too few valid v1 rows; v2's constrained decoding is always valid |
| Regression datasets default to regression (MSE); `binarize: median` gives the K = 2 version | This was the user's choice |

### v2 novelties (full math in `v2/docs/NOVELTY.md`)

| id | change | where |
|---|---|---|
| N1 | Raw/semantic serialization: real category names and units instead of MinMax codes (`serialization=encoded` switches back) | `codec.py` |
| N2 | Schema-constrained field-wise decoding. Categoricals are scored exactly over their candidates, reusing the KV cache. Numerics use FSM logit masking (sign, ≤ max integer digits, fixed decimals, no leading zeros) with canonical piece-wise tokenization | `decoding.py`, `sampler.py` |
| N3 | Weighted value-only NLL: scaffold weight 0, target weight λ = 3, terminator token counted | `codec.py`, `tabgen/llm/train_loop.py` |
| N4 | Exact label posterior averaged over R = 4 feature orders; for regression, the mean of R constrained draws | `posterior.py` |
| N5 | Curation. A DCR privacy gate (τ = α-quantile of holdout→train DCR), then cross-fitted discriminator SIR with weights (n_syn/n_real)·D/(1−D), clipped at 20, raised to β | `curation.py` |

---

## 4. Build steps as performed

1. Read the paper (PyMuPDF text extract), then checked the authors' preprocessing against the user's
   table (adult 11 features, bank 14, fuel 8).
2. Asked the four clarifying questions (§2.2).
3. Wrote the configs: `datasets.yaml` (27 datasets), `llms.yaml` (18 models with aliases),
   `paper_reference.yaml` (Table II and Table III numbers).
4. Wrote `common/tabgen` (data → llm → eval → baselines → viz → experiment → audit → smoke → cli).
5. Wrote v1 (methods, serialize, finetune, sampler, querier, generator, cli, smoke) with its config,
   tests and README.
6. Wrote v2 (codec, decoding, sampler, posterior, curation, finetune, methods, generator, cli, smoke)
   with its config, tests, README and `docs/NOVELTY.md`.
7. Wrote the docs: root `README.md`, `COMMANDS.md`, `docs/DATASETS.md`, `docs/LLM_SWITCH.md`, plus
   `pyproject.toml`, `requirements.txt` and `.gitignore`.
8. Refactored the duplicated CLI code into `tabgen/cli_common.py`. Changed `exp_dir` to take `cfg` so
   smoke runs go under `_smoke`. Gave `report.collect` and `build_tables` a `subdir` argument.
9. Ran the checks in §5 and fixed what they found (§6).
10. Added the self-contained setup guide (§0) and a detailed smoke section (§4) to `COMMANDS.md`.
11. Wrote `docs/FEASIBILITY_FR_EN.md` (static analysis only).

---

## 5. Verification actually run (no training, no downloads after prompt 3)

```bash
py -3.12 -m compileall -q common v1 v2          # OK
py -3.12 -c "import yaml,glob;[yaml.safe_load(open(f)) for f in glob.glob('**/*.yaml',recursive=True)]"   # OK
uvx ruff check common v1 v2                      # clean after --fix (2 unused imports)
py -3.12 <scratchpad>/importcheck.py             # every module imports (see note)
py -3.12 -m predllm_v1 methods                   # lists presets
py -3.12 -m predllm_v2 methods                   # lists presets
```

Import-check note: `transformers`, `peft`, `xgboost` and `sdv` were stubbed because they were not
installed. The real `torch` was imported before the stubs, and `__main__` modules were skipped so the
CLI would not run. The check also confirmed that the 27 datasets resolve, that 18 LLMs are
registered, that `resolve_llm("8b")` returns `Qwen/Qwen3-8B-Base`, and that `1.5b` maps to LoRA mode.

**Not run:** `pytest`, either smoke test, any training, SDV, or any GPU code. The first real
validation will be `python -m predllm_v1 smoke` and `python -m predllm_v2 smoke` after setup
(`COMMANDS.md` §0 and §4).

---

## 6. Problems hit and fixes

| problem | fix |
|---|---|
| `pdftoppm` missing, so the PDF could not be read | extracted the text with PyMuPDF |
| `python` not on PATH | used `py -3.12` |
| `density_coverage` built a 3-D distance tensor (~8 GB) | switched to chunked `euclidean_distances` |
| v1 start tokens used `:g` (6 significant digits), which did not match the training text | `format_number` is now used everywhere |
| Awkward code: the attention output path, an `if False else` line in `build_schema`, a no-op `rename` | rewritten |
| Duplicated argparse code in v1 and v2 | moved into `tabgen/cli_common.py` |
| The FSM test reached into internals | `NumericConstraint.run` is now a staticmethod |
| The import check ran the CLI through `__main__`, and the stubs broke torch | skipped `__main__` and imported the real torch first |
| The torch download was very slow | stopped at the user's request (prompt 3) |

---

## 7. State left on disk

* `data/raw/` holds 19 datasets downloaded before prompt 3: abalone, adult, australian, balance,
  bank, blood, breast, cardio, churn, compas, diabetes, fuel, iris, kc1, kc2, phoneme, qsar, steel,
  waveform, plus the `_openml` and `_sklearn` caches.
  Still to fetch: california, german, car, sick, insurance, gem_price, housing, credit_card.
  They download on `python -m tabgen data download --dataset all`. The Kaggle ones use a mirror or
  need `~/.kaggle/kaggle.json`; see `docs/DATASETS.md`.
* `.venv` is only partly built and has no torch. Follow `COMMANDS.md` §0.3–0.5; on this machine,
  install torch from the cu128 index.
* `v1/runs/` and `v2/runs/` are empty.

---

## 8. To replay or continue

1. Follow `COMMANDS.md` §0 (setup), then §0.7 (verify).
2. `python -m pytest -m "not smoke"`, then both smoke tests (§4). Check each
   `runs/_smoke/SMOKE_REPORT.md` for `OVERALL: PASS`.
3. Run the paper tables with `COMMANDS.md` §6 (v1) and §7 (v2), then `python -m tabgen report --versions v1,v2`.

Open items:

* The English/French minority case needs the six changes in `docs/FEASIBILITY_FR_EN.md` §5: a
  text column kind, text-aware TSTR, imbalance metrics and audits, minority-aware generation, a
  multilingual backbone, and more seeds. None of these are implemented, and the user has not yet
  said whether the data has a free-text column.

---

## 9. File-by-file specification (enough to rebuild the repo from scratch)

### 9.1 `configs/`

**`datasets.yaml`** has 27 entries. Each entry has these fields:

| field | meaning |
|---|---|
| `group` | `[paper]`, `[user]`, or both |
| `task` | classification or regression |
| `source` | one of the four types listed below |
| `target` | a column name, or `auto` |
| `numerical`, `categorical` | column lists |
| `drop`, `drop_prefix` | columns to remove |
| `binarize` | `null` by default |
| `expected` | `{n, m, k}`, checked by the data audit |

The four source types:

| source type | fields |
|---|---|
| sklearn | `loader` |
| openml | `data_id` |
| url | `urls` |
| kaggle | `slug`, `file`, optional mirror `urls` |

Sources for each dataset:

| source | datasets |
|---|---|
| sklearn | iris (`load_iris`, target `target`), breast (`load_breast_cancer`), california (`fetch_california_housing`, target `MedHouseVal`) |
| OpenML id | blood 1464, steel 1504, diabetes 37, australian 40981, balance 11, qsar 1494, phoneme 1489, waveform 60, churn 40701, kc1 1067, kc2 1063, cardio 1466, german 31, car 40975, sick 38 |
| URL: `https://raw.githubusercontent.com/nphdang/Pred-LLM/main/data/` | compas, bank, adult, abalone, fuel |
| Kaggle (+ mirror) | insurance, gem_price, housing, credit_card |

sick drops `TBG` and `TBG_measured`.

Column specs for the URL datasets:

| dataset | target | numerical | categorical |
|---|---|---|---|
| compas | two_year_recid | age, juv_fel_count, juv_misd_count, juv_other_count, priors_count | race, sex, c_charge_degree |
| bank | subscribe | balance, duration, campaign, pdays, previous | age, marital, job, education, default, housing, loan, contact, poutcome |
| adult | income-per-year | 5 columns | 6 columns |
| abalone | Age | Length, Diameter, Height, Shucked_Weight, Viscera_Weight, Shell_Weight, Rings | Sex |
| fuel | CONSUMPTION | ENGINE_SIZE, CYLINDERS, COEMISSIONS | MAKE, MODEL, VEHICLE_CLASS, TRANSMISSION, FUEL |

Column specs for the Kaggle datasets:

| dataset | slug / file | mirror | target | columns |
|---|---|---|---|---|
| insurance | `mirichoi0218/insurance` / insurance.csv | `stedy/Machine-Learning-with-R-datasets` | charges | num: age, bmi, children; cat: sex, smoker, region |
| gem_price | `colearninglounge/gemstone-price-prediction` / cubic_zirconia.csv | none | price | num: carat, depth, table, x, y, z; cat: cut, color, clarity |
| housing | `camnugent/california-housing-prices` / housing.csv | `ageron/handson-ml2` | median_house_value | 8 numerical + ocean_proximity |
| credit_card | `sakshigoyal7/credit-card-customers` / BankChurners.csv | none | Attrition_Flag | drop CLIENTNUM; `drop_prefix` Naive_Bayes_Classifier; 14 numerical + 5 categorical |

**`llms.yaml`**
- Default model: `distilgpt2`.
- `lora_defaults`: r 16, alpha 32, dropout 0.05, `target_modules: all-linear`.
- Aliases: `tiny`→tiny-gpt2, `paper`→distilgpt2, `0.5b`→qwen2.5-0.5b, `1b`→llama3.2-1b, `1.5b`→qwen2.5-1.5b, `1.7b`→qwen3-1.7b, `3b`→qwen2.5-3b, `4b`→qwen3-4b, `7b`→qwen2.5-7b, `8b`→qwen3-8b.

| key | hf_id | params | mode | min VRAM (GB) | gated |
|---|---|---|---|---|---|
| tiny-gpt2 | sshleifer/tiny-gpt2 | 0.1M | full | 0 | |
| tiny-llama | hf-internal-testing/tiny-random-LlamaForCausalLM | 1M | lora | 0 | |
| distilgpt2 | distilbert/distilgpt2 | 82M | full | 2 | |
| gpt2 | openai-community/gpt2 | 124M | full | 3 | |
| gpt2-medium | openai-community/gpt2-medium | 355M | full | 6 | |
| smollm2-360m | HuggingFaceTB/SmolLM2-360M | 360M | full | 6 | |
| qwen2.5-0.5b | Qwen/Qwen2.5-0.5B | 0.5B | lora | 3 | |
| qwen3-0.6b | Qwen/Qwen3-0.6B-Base | 0.6B | lora | 3 | |
| llama3.2-1b | meta-llama/Llama-3.2-1B | 1.2B | lora | 4 | yes |
| qwen2.5-1.5b | Qwen/Qwen2.5-1.5B | 1.5B | lora | 5 | |
| qwen3-1.7b | Qwen/Qwen3-1.7B-Base | 1.7B | lora | 6 | |
| smollm2-1.7b | HuggingFaceTB/SmolLM2-1.7B | 1.7B | lora | 6 | |
| qwen2.5-3b | Qwen/Qwen2.5-3B | 3.1B | qlora | 5 | |
| llama3.2-3b | meta-llama/Llama-3.2-3B | 3.2B | qlora | 5 | yes |
| qwen3-4b | Qwen/Qwen3-4B-Base | 4.0B | qlora | 6 | |
| qwen2.5-7b | Qwen/Qwen2.5-7B | 7.6B | qlora | 10 | |
| llama3.1-8b | meta-llama/Llama-3.1-8B | 8.0B | qlora | 12 | yes |
| qwen3-8b | Qwen/Qwen3-8B-Base | 8.2B | qlora | 12 | |

**`paper_reference.yaml`**
- Table II, with columns original, copulagan, ctgan, tvae, great, taptap, pred_llm:
  - 17 accuracy datasets;
  - MSE for abalone, fuel and california.
- Table III: great 0.6363, taptap 0.8372, abl_x_target_none 0.6191, abl_xy_feature_none 0.7352, abl_xy_target_llm 0.7307, abl_xy_feature_llm 0.7477, abl_x_target_llm 0.8495, pred_llm 0.8658.

### 9.2 `common/tabgen/` (shared library)

| module | contents |
|---|---|
| `__init__.py` | module-map docstring, `__version__ = "1.0.0"` |
| `paths.py` | `ROOT` (env `PALT_ROOT` or `parents[2]`), `CONFIGS`, `DATA_RAW` = ROOT/data/raw, `REPORTS`; `version_root(v)`, `runs_root(v)` |
| `utils.py` | `set_seed`, `load_yaml`, `deep_update`, `apply_overrides(cfg, ["a.b=val"])` (values parsed as YAML), numpy-safe `save_json`/`load_json`, `get_logger`, `pick_device`, `Timer`, `format_number(v, decimals)` (rounds, strips trailing zeros, never prints "-0") |
| `data/registry.py` | `DatasetSpec`; `load_registry` (lru_cache); `get_spec`; `resolve_datasets` expands all/paper/user/classification/regression/comma lists |
| `data/download.py` | `ManualDownloadRequired`; `raw_path`, `meta_path`, `is_available`; `_save` writes csv + meta.json {target, numerical, categorical}; `_fetch_sklearn`; `_fetch_openml` (`data_home=data/raw/_openml`, types from dtypes); `_fetch_urls`; `_fetch_kaggle` (manual file → mirrors → kaggle CLI if `~/.kaggle/kaggle.json` → raise with instructions); `fetch(name, force)`; `read_raw(name)` |
| `data/preprocess.py` | see below |
| `llm/registry.py` | `LLMSpec(key, hf_id, params, mode, min_vram_gb, gated, note, lora)` with `.tag`; `list_llms()`; `resolve_llm(name, mode)` accepts an alias, a key, or a raw HF id (defaults to lora) |
| `llm/loader.py` | see below |
| `llm/train_loop.py` | see below |
| `eval/tstr.py` | `xgb_model(task, seed)` (default params); `tstr(...)`: accuracy + f1_macro, paper rule missing class → 0; regression mse/rmse/mae/r2; `primary_metric` |
| `eval/fidelity.py` | `discriminator_score` (XGB, 70/30, 2·max(AUC, 0.5)−1, lower is better); `inverse_kl` (10-bin, synthcity style); `density_coverage` (Naeem et al., k=2, chunked `euclidean_distances`); `dcr`; `fidelity_report(real, fake, features, discrete, seed, max_rows=5000)` adds dcr_mean, dcr_median, exact_copy_rate |
| `eval/report.py` | `collect(versions, subdir=None)` globs `runs_root/*/*/seed*/metrics.json`; `build_tables(versions, out_dir, subdir=None)` writes markdown + csv with the paper columns plus `fidelity.md` (needs `tabulate`) |
| `baselines/sdv_models.py` | `SDV_METHODS=("ctgan","tvae","copulagan")`; `sdv_fit_sample(method, train_clean, num_cols, cat_cols, target, task, n, epochs=300, seed, use_gpu)`; falls back from `Metadata` to `SingleTableMetadata`, and from the `enable_gpu` kwarg to `cuda` |
| `experiment.py` | see below |
| `viz/data_plots.py` | `plot_dataset(name, stages, out)` → `reports/data_viz/<name>/`: raw_missing.png, raw_distributions.png, clean_distributions.png, encoded_distributions.png, encoded_correlation.png, serialized_examples.txt (v1 and v2 text) |
| `viz/synth_plots.py` | marginals, correlation_diff, tsne (Fig 8), dcr_hist (Fig 6), feature_class_importance (Fig 2 via XGB `pred_contribs`); `plot_run(run_dir, split, fake)` |
| `viz/attention.py` | `field_attention(model, tok, text, fields, target_field, n_layers=2)` sums attention from the target-value token per field, using offset mapping; `plot_attention_comparison(results, path)` |
| `audit.py` | see below |
| `cli.py` / `__main__.py` | `python -m tabgen data {download,info,list} · viz {data,run} · audit {env,data,run} · llm {list,check} · report --versions` |
| `cli_common.py` | see below |
| `smoke.py` | see below |

**`data/preprocess.py`**
- `TARGET="target"`.
- `CleanTable(name, task, df, features, num_cols, cat_cols, target, report)`.
- `clean_table(name, binarize)` does, in order:
  1. drop the configured columns;
  2. infer num/cat, ordered num then cat when explicit lists are given;
  3. strip strings; `'?'` → NA; dropna;
  4. categoricals → str, with `,` replaced by `;`; numerics → float;
  5. regression with `binarize: median` → classes "0"/"1".
- `TablePreprocessor(num_cols, cat_cols, features, target, task, decimals=2)`:
  - `fit`: categories → codes, MinMax on every feature;
  - `transform`: rounds to 2 decimals and drops unknown-target rows;
  - `inverse_transform`, `to_dict`, `from_dict`.
- `TabularSplit` holds name, task, seed, features, num_cols, cat_cols, target, raw_train, raw_test, train, test, pre and report, plus `n_classes` and `summary()`.
- `make_split(name, seed, test_size, train_size, fit_scaler_on="full", binarize, max_rows, decimals)` makes a stratified split.

**`llm/loader.py`**
- `load_tokenizer` sets pad = eos. `vram_gb` reports GPU memory.
- `load_llm(spec, device, trainable, attn, gradient_checkpointing)`:
  - full mode loads fp32;
  - lora and qlora load bf16 on CUDA;
  - qlora uses `BitsAndBytesConfig` (nf4, double quant) and `device_map {"":0}`;
  - `_attach_lora` runs `prepare_model_for_kbit_training`, then `LoraConfig`.
- `save_finetuned(...)` writes `llm_meta.json`.
- `load_finetuned(model_dir, device, attn)` returns (tok, model, meta), wrapping adapters in `PeftModel`.

**`llm/train_loop.py`**
- `make_collate(pad_id)`: right pad, attention mask, weights.
- `weighted_lm_loss` = Σ w·nll / Σ w over the shifted targets.
- `train_lm(model, tok, dataset, cfg, out_dir, device)`:
  - AdamW with linear warmup;
  - micro-batches with gradient accumulation;
  - bf16 autocast and gradient clipping;
  - `max_steps`;
  - writes `train_log.jsonl` and returns stats.

**`experiment.py`**
- `NON_LLM = ("original",) + SDV_METHODS`.
- `exp_dir(version, dataset, method, llm_tag, cfg)` → `runs_root[/runs_subdir]/<ds>[_bin]/<method>__<llm|none>`.
- `split_from_cfg(dataset, seed, cfg)`.
- `evaluate(split, enc, cfg, seed)` returns TSTR, original, n_synthetic and fidelity.
- `run_seed`:
  - skips seeds that already have `metrics.json` unless overwrite;
  - writes config/split/preprocessor json, `synthetic_encoded.csv`, `synthetic_clean.csv`, `generation_stats.json`, `metrics.json`, and plots.
- `summarize` and `run_experiment` write `summary.json`.
- Generator contract: `make_generator(method, cfg, split, run_dir, seed)` returns an object with:
  - `.fit()` → dict;
  - `.generate(n)` → (enc_df, clean_df | None, stats).

**`audit.py`**
- `REQUIRED_RUN_FILES` and a `Report` class.
- `audit_env`: packages, CUDA, which LLMs fit.
- `audit_data(name)`:
  - size vs expected ±2%;
  - classes and balance (`counts.min() >= 0.05`);
  - constant columns, duplicates;
  - train/test overlap;
  - leakage (|corr| > 0.98).
- `audit_run(run_dir)` writes `audit.json`. Checks:
  - artifacts present, no NaN, |D_fake|;
  - values within [0, 1];
  - class coverage (FAIL if a class is missing);
  - label TV < 0.15;
  - exact copies < 0.2;
  - TSTR valid; validity rate.

**`cli_common.py`**
- `COMMANDS = [run, benchmark, ablation, train, generate, infer, evaluate, smoke, methods]`.
- `build_parser(prog, doc, extra_commands)` defines these flags:
  - `--dataset`, `--datasets`, `--method`, `--methods`;
  - `--llm`, `--mode`, `--device`, `--config`;
  - `--runs`, `--seeds`, `--epochs`, `--max-steps`;
  - `--binarize`, `--max-rows`, `--set`;
  - `--run`, `--n`, `--out`, `--plots`, `--overwrite`.
- `build_cfg`, `seeds_of`.

**`smoke.py`**
- `run_smoke(version, cfg, make_generator, methods, llm_tag, dataset="iris")`.
- Sets `runs_subdir="_smoke"` and skips SDV methods when sdv is missing.
- Checks that the required files exist, plus `marginals.png`, `tsne.png` and `dcr.png`; LLM methods also need `model_ref.json` and `train_log.jsonl`.
- Runs `audit_run`, builds tables into `runs/_smoke/tables`, and writes `SMOKE_REPORT.md` ending in `OVERALL: PASS|FAIL`. Returns 0 or 1.

**`common/tests/test_common.py`**: `format_number` cases, the iris split protocol, the categorical round trip, tstr/fidelity, the LLM switch, and overrides.

### 9.3 `v1/predllm_v1/` (replication)

**`methods.py`**
- `MethodSpec(name, permutation, sampling, query, augment_original, paper_row)`.
- `resolve_methods`.
- Presets:

| preset | permutation | sampling | query | augment_original |
|---|---|---|---|---|
| great | xy | target | none | False |
| taptap | xy | target | classifier | False |
| abl_x_target_none | x | target | none | True |
| abl_xy_feature_none | xy | feature | none | True |
| abl_xy_target_llm | xy | target | llm | True |
| abl_xy_feature_llm | xy | feature | llm | True |
| abl_x_target_llm | x | target | llm | True |
| pred_llm | x | feature | llm | True |

- Groups:

| group | members |
|---|---|
| llm | the LLM presets |
| ablation | the Table III presets |
| sdv | ctgan, tvae, copulagan |
| paper | original, copulagan, ctgan, tvae, great, taptap, pred_llm |

**`serialize.py`**
- `row_fields(row, features, task, decimals=2, target_decimals=4)`.
- `column_order(features, strategy, rng, original, with_target)`: `permute_x` shuffles the features and keeps the target last; `permute_xy` shuffles everything.
- `encode(fields, order)` produces `"X is v, …, target is y"`.
- `parse(text, columns)` takes the first float per known column.

**`finetune.py`**: `RowTextDataset(fields, features, tok, permutation, augment_original, seed)`.
- Length is 2N when augmenting; the second half uses the original order (D'_real = permuted ∪ original).
- Appends EOS. `.text(i)`.

**`sampler.py`**
- `generate_text(model, tok, prompts, max_new_tokens, temperature, top_k, batch_size)`, with left padding.
- `_fmt_start` uses `format_number`.
- `sample_rows(model, tok, train, features, task, n, strategy, require_target, cfg, rng)` returns (df, stats with validity_rate).
- Feature strategy: n/M rows per feature, each prompt starts `"Xi is v"` with v ~ empirical p(Xi).

**`querier.py`**
- `query_llm(model, tok, X, features, cfg, fallback)` builds an original-order prompt ending `", target is"` (when `force_target_key`), with retries and a fallback.
- `query_classifier(train, X, features, task, seed)` (TapTap).

**`generator.py`**
- `VERSION="v1"`.
- `finetune_cfg(cfg, spec)`: LoRA uses `lr_lora` and micro-batch 8 or 4.
- `PredLLMV1`:
  - `fingerprint()`;
  - `model_dir()` → `run_dir.parent.parent/_models/<tag>_<fp>`;
  - `fit()` reuses a cached model or trains and saves; writes `model_ref.json` and copies `train_log`;
  - `_set_lengths()`: `max_new_tokens` = longest row in tokens + 8; query length 16;
  - `generate(n)`: sample → query → clip ±clip → round.
- `make_generator`, `load_generator(run_dir)`.

**`cli.py`**
- Standard commands, plus:
  - `infer`: LLM as predictor → `infer_llm_as_predictor.json`;
  - `attention`: eager attention → `plots/attention.png`, `attention.json`.
- Smoke defaults to distilgpt2 with `max_steps` 150.

**`smoke.py`**
- `run_smoke(llm="distilgpt2", device="auto", max_steps=150)`; tiny-gpt2 is remapped to distilgpt2.
- Overrides: `finetune.reuse=false`, `eval.runs=1`, `baselines.epochs=5`, `log_every=10`.
- Methods: original, pred_llm, great, taptap, ctgan.

**`v1/configs/default.yaml`**

| section | values |
|---|---|
| top level | llm distilgpt2, mode auto, device auto |
| data | test_size 0.2, train_size 1.0, gen_size 1.0, fit_scaler_on full, decimals 2, binarize null, max_rows null |
| finetune | epochs 50, batch_size 32, micro_batch_size null, lr 5e-5, lr_lora 2e-4, weight_decay 0, warmup 0, max_grad_norm 1, max_steps null, log_every 50, reuse true |
| sampling | temperature 0.7, top_k 50, batch_size 100, max_rounds 30, target_decimals 4, clip 10 |
| query | max_retries 5, force_target_key true |
| eval | runs 3, fidelity true, max_fidelity_rows 5000 |
| baselines | epochs 300 |

**Tests**
- `test_v1_units.py`: permutation, encode/parse round trip, garbage parsing, dataset size and order (CharTok), presets.
- `test_v1_smoke.py`: `@pytest.mark.smoke`, asserts `run_smoke(max_steps=150) == 0` and that the report contains `OVERALL: PASS`.

### 9.4 `v2/predllm_v2/` (novelty)

**`codec.py`**
- `Schema(features, target, task, kinds, decimals, ranges, cats)`, with `columns`, `fmt` and `numeric_spec` → (max_int, decimals, allow_neg).
- `build_schema(split, serialization, max_decimals=4)`:
  - raw serialization → `split.raw_train` and the real target name;
  - encoded serialization → `split.train` and `"target"`.
- `FieldCodec(schema, tok)`:
  - BOS = `tok("", add_special_tokens=True)`; the separator must be a single token;
  - cached `key_ids(col, first)` and `val_ids(s)`;
  - `terminator`;
  - `encode(row, order, lam, w_scaffold)` → ids, weights;
  - `prefix(row, order, then_key)`.

**`decoding.py`**
- `left_pad` returns ids, mask and positions. `_expand_cache` does a deepcopy plus `batch_repeat_interleave`, with a tuple fallback.
- `score_continuations(model, prefixes, cands, pad_id, max_elems=3e7, use_cache=True)` → (B, K). Chunk size: `per = max(1, max_elems // (B*(Lc-1)*V))`. `_score_no_cache` is the reference implementation.
- `NumericConstraint(tok, vocab_size, device)`:
  - a surface map of numeric-character tokens;
  - staticmethods `advance`, `run`, `accepting`, with `START=(0,0,0,False,0)`;
  - phases 0–4, at most 2 leading spaces, no leading zeros;
  - `mask(st, spec, terminator)` is cached.
- `decode_numeric(...)` uses the KV cache and explicit `position_ids`; `max_steps` = max_int + frac + 4.

**`sampler.py`**
- `sample_categorical` uses sequence-level T.
- `sample_numeric` clips to the observed range and falls back to the empirical distribution.
- `sample_features(...)`:
  - quota ceil(n/M) rows per conditioning feature;
  - order = [c0] + the rest shuffled; the first value is drawn from the empirical distribution;
  - stats include `numeric_fallbacks` (validity is 1.0).

**`posterior.py`**
- `_orders`.
- `class_posterior(model, codec, rows, R, cfg, rng)` returns the mean softmax over R orders.
- `decide_labels(probs, how, temperature, rng)`.
- `regression_values`: the mean of R draws.

**`curation.py`**
- `privacy_gate(real, pool, alpha, max_drop, seed)`: a 90/10 internal split sets τ.
- `sir_weights(real, pool, beta, clip, folds, seed)`: StratifiedKFold XGB out-of-fold predictions; reports ESS.
- `curate(real_enc, pool_enc, n, features, cfg, seed)` → (selected, idx, keep, stats).

**`finetune.py`**: `SchemaRowDataset(rows, codec, lam, w_scaffold, perm_copies=1, orig_copies=1, seed)` yields {input_ids, weights}.

**`methods.py`**
- Presets are lists of config overrides:

| preset | overrides |
|---|---|
| pred_llm_v2 | none |
| v2_no_curation | `curation.enabled=false` |
| v2_R1 | `query.permutations=1` |
| v2_argmax | `query.label_decode=argmax` |
| v2_plain_loss | `loss.target_weight=1.0`, `loss.scaffold_weight=1.0` |
| v2_encoded | `serialization=encoded` |

- Groups: v2, ablation, compare (original, pred_llm_v2, ctgan, tvae).

**`generator.py`**
- `PredLLMV2` applies the preset overrides, builds the schema and `train_text`, and implements `_ft_cfg` and `fingerprint` (which includes serialization and loss).
- `fit` writes `model_ref.json` and `schema.json`.
- `_ready` builds the codec and a `NumericConstraint` sized to the output embedding.
- `generate(n)`: pool → x → posterior y → frames → curate → stats.

**`cli.py`**
- `infer` reports posterior-argmax accuracy (classification) or MSE (regression).
- Smoke defaults to tiny-gpt2 with `max_steps` 30.

**`smoke.py`**
- `run_smoke(llm="tiny-gpt2", device="auto", max_steps=30)` with `query.permutations=2`.
- Methods: original, pred_llm_v2, v2_encoded, v2_plain_loss.

**`v2/configs/default.yaml`**

| section | values |
|---|---|
| serialization | raw |
| data | as v1, plus max_value_decimals 4 |
| finetune | as v1, plus warmup_ratio 0.03, perm_copies 1, orig_copies 1 |
| loss | target_weight 3.0, scaffold_weight 0.0 |
| decoding | batch_size 64, temperature_num 0.7, temperature_cat 1.0, kv_cache true, max_score_elems 3e7 |
| query | permutations 4, label_decode sample, temperature_y 0.7, reg_aggregate mean |
| curation | enabled true, pool_factor 1.5, sir true, beta 1.0, clip 20, folds 5, privacy_gate true, privacy_alpha 0.01, max_drop 0.5 |
| eval, baselines | as v1 |

**Tests**
- `test_v2_units.py`:
  - parametrized FSM cases via `NumericConstraint.run(NumericConstraint.START, s, spec)`;
  - codec weights (CharTok);
  - privacy gate and SIR on toy data;
  - a smoke-marked tiny-gpt2 test comparing cached vs no-cache scoring and checking numeric validity.
- `test_v2_smoke.py` asserts `run_smoke(llm="tiny-gpt2", max_steps=20) == 0`.

### 9.5 Docs and packaging

| file | contents |
|---|---|
| `README.md` | overview, repo tree, a "Where the LLM is used" table, quick start |
| `COMMANDS.md` | see the list below |
| `docs/DATASETS.md` | automatic, Kaggle and manual download paths; cleaning and encoding; notes (car is all-categorical, german splits 7/13, sick is smaller after dropna, `binarize` for regression) |
| `docs/LLM_SWITCH.md` | compatibility table and why only these models, the three modes, 6 GB guidance |
| `v1/README.md` | module table, protocol, deviations (EOS stop, force_target_key, SDV on the clean space, TapTap as in Table III), outputs |
| `v2/README.md`, `v2/docs/NOVELTY.md` | module table; N1–N5 math, summary table, cost, how to verify |
| `docs/FEASIBILITY_FR_EN.md` | English/French 5% minority analysis (see §10) |
| `pyproject.toml` | see the list below |
| `requirements.txt`, `.gitignore` | dependencies; ignores data/raw, runs, .venv, caches |

`COMMANDS.md` sections:
- §0 self-contained setup:
  - 0.1 prerequisites; 0.2 cd; 0.3 venv on Windows and Linux (with the execution-policy workaround);
  - 0.4 torch index table (cu128 / cu126 / cpu); 0.5 requirements + `-e .`;
  - 0.6 optional HF_HOME, login for gated models, Kaggle, HF_HUB_OFFLINE;
  - 0.7 verify; 0.8 troubleshooting.
- §1 data; §2 data plots; §3 LLM switch.
- §4 smoke tests: commands, comparison table, pass criteria, outputs, variants, troubleshooting.
- §5 tests; §6 v1 training; §7 v2 training; §8 inference; §9 result plots; §10 audits; §11 report.

`pyproject.toml`:
- setuptools, finding packages in `common`, `v1` and `v2`.
- Dependencies: torch, transformers ≥ 4.51, peft ≥ 0.13, accelerate, xgboost, scikit-learn, pandas, numpy, scipy, pyyaml, matplotlib, tabulate, tqdm.
- Extras: qlora, baselines, kaggle, dev.
- pytest: testpaths, `--import-mode=importlib`, a `smoke` marker.

---

## 10. Feasibility answer given (English/French, French = 5%)

- **Free-text column:** not feasible with v1 or v2 as they stand.
  - v1 turns each sentence into a category code, so neither the LLM nor XGBoost ever sees the words.
  - v2 can only replay training sentences, and for N = 5,000 that costs about 375k forward passes.
  - Estimated as-is: accuracy ~0.95 (misleading), French recall ~0–0.3, macro-F1 ~0.5–0.6.
  - After the six changes with `--llm 1.5b`: French recall ~0.85–0.98, macro-F1 ~0.9–0.98. All of these are estimates.
- **No free text:** feasible; needs the imbalance fixes only.
- **Supporting facts:**
  - T = 0.7 turns p = 0.05 into 0.015.
  - 95% CI half-width of French recall: ±0.19 at N = 1k, ±0.08 at 5k, ±0.04 at 20k.
  - The TV < 0.15 audit misses a 5%→1% collapse.
  - Aim for N ≳ 5,000 (≥ 200 French training rows).
- **Still unanswered by the user:**
  - Does the data have a free-text column?
  - Should the six changes be implemented in v2?

---

## 11. Full transcript

The complete session transcript, with every tool call and file write, is at
`C:\Users\Gaurav\.claude\projects\D--CODING-Resume\e07119f4-4683-4bf8-95d4-670545ea2127.jsonl`.
