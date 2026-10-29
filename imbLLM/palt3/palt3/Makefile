.DEFAULT_GOAL := help
SHELL := /bin/bash

# Common variables (override on the command line: make run-paper MODEL=qwen2.5-1.5b)
DATASETS ?= fast
SEEDS    ?= 0 1 2 3 4 5 6 7 8 9   # 10 re-splits (docs/RESEARCH_REVIEW.md G2)
MODEL    ?= distilgpt2
HW       ?= auto
CONFIG   ?= paper
DS       ?= australian
METRIC   ?= f1
RUN_ARGS  = --datasets $(DATASETS) --seeds $(SEEDS) --model $(MODEL) --hardware $(HW) --config $(CONFIG)

# ---------------------------------------------------------------- install
.PHONY: install install-llm install-all
install:        ## core: data, audits, classical baselines, evaluation
	pip install -e ".[dev]"
install-llm:    ## + torch/transformers/peft (any LLM method)
	pip install -e ".[llm,dev]"
install-all:    ## + 4-bit QLoRA, CTGAN/TVAE, Kaggle   (flash-attn separately: make install-flash)
	pip install -e ".[llm,quant,sdv,kaggle,dev]"
install-synthcity: ## TabDDPM / ARF baselines (heavy; separate env if torch versions clash)
	pip install -e ".[synthcity,lgbm]"
install-flash:  ## FlashAttention-2 for A100/H100 (needs a CUDA toolchain)
	pip install -e ".[flash]" --no-build-isolation

# ---------------------------------------------------------------- data
.PHONY: data-list data-import data-fetch data-verify data-inspect viz-data
data-list:      ## registry: 28 datasets, sources, groups, on-disk status
	imb data list
data-import:    ## copy the 19 csvs palt/ already downloaded (offline)
	imb data fetch --datasets all --from ../palt/data/raw || true
data-fetch:     ## download everything reachable without credentials
	imb data fetch --datasets open
data-verify:    ## shapes vs the papers' tables
	imb data verify --datasets all
data-inspect:   ## splits / imbalance / types for one dataset  [DS=]
	imb data inspect --dataset $(DS)
viz-data:       ## before/after cleaning + protocol + minority geometry figures  [DATASETS=]
	imb viz data --datasets $(DATASETS)

# ---------------------------------------------------------------- checks
.PHONY: test test-llm test-slow audit smoke check
test:           ## fast tests: no torch, no network (~10 s)
	pytest -q -m "not llm and not slow"
test-llm:       ## + end-to-end LLM tests on the offline tiny backbone (minutes)
	pytest -q -m "not slow"
audit:          ## math (shipped code) + paper tables + data checks
	imb audit all
audit-contamination: ## memorisation probes on the pre-trained MODEL (loads it)  [MODEL=]
	imb audit contamination --model $(MODEL) --datasets adult,german,bank,compas
smoke:          ## every code path on the offline tiny-random backbone
	imb smoke
check: test audit   ## what CI should run

# ---------------------------------------------------------------- LLM switch
.PHONY: models plan hw pull where-llm
models:         ## backbones x hardware profiles (mode/precision each would use)
	imb models
plan:           ## how MODEL runs on HW   [MODEL= HW=]
	imb plan --model $(MODEL) --hardware $(HW)
hw:             ## detected device + profiles
	imb hw
pull:           ## pre-download MODEL weights (else downloaded on first use)
	imb pull --model $(MODEL)
where-llm:      ## every place the LLM is used
	imb where-llm --grep

# ---------------------------------------------------------------- experiments
.PHONY: run-baselines run-paper run-llm-baselines run-ablation run-v2 run-flags \
        run-interp run-tau run-temperature run-gibbs run-prior run-qsweep run-rsweep         run-order run-curation
run-baselines:      ## Table II non-LLM columns + threshold/prior-shift references
	imb run baselines --datasets $(DATASETS) --seeds $(SEEDS) --resume
run-paper:          ## Table II ImbLLM row (v1_paper)
	imb run paper $(RUN_ARGS) --resume
run-llm-baselines:  ## GReaT / TapTap / Pred-LLM style
	imb run llm_baselines $(RUN_ARGS) --resume
run-ablation:       ## Table III + IV
	imb run ablation $(RUN_ARGS) --resume
run-v2:             ## headline: v1_exact vs recommended vs full
	imb run v2_compare $(RUN_ARGS) --resume
run-flags:          ## each v2 flag added to v1 / removed from recommended
	imb run v2_flags $(RUN_ARGS) --resume
run-interp:         ## 5 interpolation geometries
	imb run interp $(RUN_ARGS) --resume
run-tau:            ## tau sweep (one fine-tune per dataset-seed, cached)
	imb run tau $(RUN_ARGS) --resume
run-temperature:    ## temperature sweep (cached fine-tune)
	imb run temperature $(RUN_ARGS) --resume
run-gibbs:          ## order sensitivity + SIR-Gibbs sweeps
	imb run gibbs $(RUN_ARGS) --resume
run-prior:          ## P8: pretrained/random x named/anonymised
	imb run prior2x2 $(RUN_ARGS) --resume
run-qsweep:         ## Fig. 6
	imb run qsweep $(RUN_ARGS) --resume
run-rsweep:         ## Fig. 5
	imb run rsweep $(RUN_ARGS) --resume
run-order:          ## order sensitivity: permuted vs fixed-order fine-tune, real-row reference
	imb run order $(RUN_ARGS) --resume
run-curation:       ## {ImbLLM, v2, SMOTE-NC} x {raw, CLLM-curated}
	imb run curation $(RUN_ARGS) --resume

# ---------------------------------------------------------------- reports
.PHONY: report
report:         ## EXP=<experiment> METRIC=f1|real_f1|auc|auprc|f1_tuned|...
	imb report $(EXP) --metric $(METRIC) --figure

.PHONY: clean-models
clean-models:   ## delete cached fine-tuned models
	imb clean --models

.PHONY: help
help:
	@echo "palt3 / imbllm -- ImbLLM replication (v1_paper) + corrected extension (v2_corrected)"
	@echo "vars: DATASETS=$(DATASETS) SEEDS='$(SEEDS)' MODEL=$(MODEL) HW=$(HW) CONFIG=$(CONFIG)"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n",$$1,$$2}'
