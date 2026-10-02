# palt: LLM tabular data generation (Pred-LLM v1 → v2)

This monorepo holds two versions:

* **v1** is a faithful replication of *Generating Realistic Tabular Data with Large Language Models*
  (Pred-LLM, arXiv:2410.21717). It includes GReaT, TapTap and every Table III ablation as switches
  on one pipeline, plus CTGAN / TVAE / CopulaGAN.
* **v2** keeps Pred-LLM's ideas and adds schema-constrained decoding, exact permutation-marginalized
  labels, a weighted value-only loss, semantic serialization and discriminator-guided curation.
  See [v2/docs/NOVELTY.md](v2/docs/NOVELTY.md).

It covers 27 datasets (the paper's 20 plus the extended 10) and one LLM switch from 82M to 8B
parameters (full / LoRA / QLoRA).

**All commands are in [COMMANDS.md](COMMANDS.md).**

```
palt/
├── configs/            datasets.yaml (27 datasets) · llms.yaml (LLM switch) · paper_reference.yaml (Table II/III)
├── docs/               DATASETS.md (download guide) · LLM_SWITCH.md (compatible LLMs)
├── common/tabgen/      shared library used by both versions
│   ├── data/           registry · download · cleaning + paper encoding + splits
│   ├── llm/            registry (switch) · loader (full/LoRA/QLoRA) · weighted training loop
│   ├── eval/           TSTR (XGBoost) · fidelity (discriminator, inverse KL, density, coverage, DCR) · tables
│   ├── baselines/      SDV CTGAN / TVAE / CopulaGAN
│   ├── viz/            data-stage plots · synthetic-vs-real plots · attention (Fig. 4)
│   ├── experiment.py   version-agnostic runner  -> <version>/runs/...
│   ├── audit.py        env / data / run audits
│   ├── smoke.py        shared smoke harness
│   └── cli.py          python -m tabgen {data,viz,audit,llm,report}
├── v1/predllm_v1/      replication  (README.md explains every module)   configs/ · tests/ · runs/
├── v2/predllm_v2/      improved method (README.md, docs/NOVELTY.md)     configs/ · tests/ · runs/
├── data/raw/           downloaded datasets (git-ignored)
└── reports/            data plots, audits, result tables
```

## Where the LLM is used

| phase | v1 | v2 |
|---|---|---|
| load / switch | `tabgen/llm/loader.py` (the only place a pretrained LLM is instantiated) | same |
| fine-tuning | `tabgen/llm/train_loop.py` via `predllm_v1/generator.py::fit` | same loop with per-token weights via `predllm_v2/generator.py::fit` |
| sampling x | `predllm_v1/sampler.py` (free-text generation) | `predllm_v2/sampler.py` + `decoding.py` (exact scoring / FSM) |
| label y | `predllm_v1/querier.py::query_llm` (prompting) | `predllm_v2/posterior.py` (permutation-marginalized posterior) |
| inference commands | `predllm_v1 generate / infer / attention` | `predllm_v2 generate / infer` |
| diagnostics | `tabgen llm check`, `tabgen/viz/attention.py` | same |

Everything else (data, XGBoost evaluation, SDV baselines, metrics, curation, plots, audits) does not
use the LLM.

## Quick start
```bash
.venv\Scripts\activate                     # Windows  (source .venv/bin/activate on Linux)
python -m tabgen audit env
python -m tabgen data download --dataset all
python -m pytest -m "not smoke"
python -m predllm_v1 smoke && python -m predllm_v2 smoke
python -m predllm_v1 run --dataset iris --method pred_llm
python -m predllm_v2 run --dataset iris --method pred_llm_v2
python -m tabgen report --versions v1,v2
```
