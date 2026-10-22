.DEFAULT_GOAL := help
SHELL := /bin/bash

# Common variables (override on the command line: make run-paper MODEL=qwen2.5-1.5b)
DATASETS ?= natural_fast
SEEDS    ?= 0 1 2 3 4 5 6 7 8 9   # 10 re-splits (docs/RESEARCH_REVIEW.md G2)
MODEL    ?= qwen3-1.7b
HW       ?= auto
CONFIG   ?= laptop_dev
DS       ?= pc1
METRIC   ?= gain_vs_threshold
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
data-list:      ## registry: 41 entries (natural, paper, predllm, ...), sources, on-disk status
	tilt data list
data-import:    ## copy the tables palt3/ already has (offline)
	tilt data fetch --datasets all --from ../palt3/data/raw || true
data-fetch:     ## download everything reachable without credentials
	tilt data fetch --datasets open
data-verify:    ## shapes vs the papers' tables
	tilt data verify --datasets all
data-inspect:   ## splits / imbalance / types for one dataset  [DS=]
	tilt data inspect --dataset $(DS)
viz-data:       ## before/after cleaning + protocol + minority geometry figures  [DATASETS=]
	tilt viz data --datasets $(DATASETS)

# ---------------------------------------------------------------- checks
.PHONY: test test-llm test-slow audit smoke check
test:           ## fast tests: no torch, no network (~10 s)
	pytest -q -m "not llm and not slow"
test-llm:       ## + end-to-end LLM tests on the offline tiny backbone (minutes)
	pytest -q -m "not slow"
audit:          ## theory + math (shipped code) + paper tables + data checks
	tilt audit all
theory:         ## the thesis propositions, numerically (CPU, ~15 s)
	tilt audit theory -v
oracle:         ## E0: no-LLM study, does tilting help a PERFECT generator? (CPU)
	tilt oracle --n-min 20 50 100 --seeds 30 --learners lr xgb
audit-contamination: ## memorisation probes on the pre-trained MODEL (loads it)  [MODEL=]
	tilt audit contamination --model $(MODEL) --datasets adult,german,bank,compas
smoke:          ## every code path on the offline tiny-random backbone
	tilt smoke
check: test audit   ## what CI should run

# ---------------------------------------------------------------- LLM switch
.PHONY: models plan hw pull where-llm
models:         ## backbones x hardware profiles (mode/precision each would use)
	tilt models
plan:           ## how MODEL runs on HW   [MODEL= HW=]
	tilt plan --model $(MODEL) --hardware $(HW)
hw:             ## detected device + profiles
	tilt hw
pull:           ## pre-download MODEL weights (else downloaded on first use)
	tilt pull --model $(MODEL)
where-llm:      ## every place the LLM is used
	tilt where-llm --grep

# ---------------------------------------------------------------- experiments
.PHONY: run-baselines run-paper run-llm-baselines run-ablation run-v2 run-flags \
        run-interp run-tau run-temperature run-gibbs run-prior run-qsweep run-rsweep         run-order run-curation run-main run-utility run-budget run-sampler run-validity         run-scale run-scarcity run-replication
run-baselines:      ## Table II non-LLM columns + threshold/prior-shift references
	tilt run baselines --datasets $(DATASETS) --seeds $(SEEDS) --resume
run-paper:          ## Table II ImbLLM row (v1_paper)
	tilt run paper $(RUN_ARGS) --resume
run-llm-baselines:  ## GReaT / TapTap / Pred-LLM style
	tilt run llm_baselines $(RUN_ARGS) --resume
run-ablation:       ## Table III + IV
	tilt run ablation $(RUN_ARGS) --resume
run-v2:             ## headline: v1_exact vs recommended vs full
	tilt run v2_compare $(RUN_ARGS) --resume
run-flags:          ## each v2 flag added to v1 / removed from recommended
	tilt run v2_flags $(RUN_ARGS) --resume
run-interp:         ## 5 interpolation geometries
	tilt run interp $(RUN_ARGS) --resume
run-tau:            ## tau sweep (one fine-tune per dataset-seed, cached)
	tilt run tau $(RUN_ARGS) --resume
run-temperature:    ## temperature sweep (cached fine-tune)
	tilt run temperature $(RUN_ARGS) --resume
run-gibbs:          ## order sensitivity + SIR-Gibbs sweeps
	tilt run gibbs $(RUN_ARGS) --resume
run-prior:          ## P8: pretrained/random x named/anonymised
	tilt run prior2x2 $(RUN_ARGS) --resume
run-qsweep:         ## Fig. 6
	tilt run qsweep $(RUN_ARGS) --resume
run-rsweep:         ## Fig. 5
	tilt run rsweep $(RUN_ARGS) --resume
run-order:          ## order sensitivity: permuted vs fixed-order fine-tune, real-row reference
	tilt run order $(RUN_ARGS) --resume
run-curation:       ## {ImbLLM, v2, SMOTE-NC} x {raw, CLLM-curated}
	tilt run curation $(RUN_ARGS) --resume

# ---------------------------------------------------------------- TILT (palt4)
run-main:           ## E1 headline: TILT vs threshold / SMOTE / ImbLLM / v2 / CLLM  [CONFIG=h100_main]
	tilt run thesis_main --config $(CONFIG) --resume
run-utility:        ## E2 which utility
	tilt run utility --config $(CONFIG) --resume
run-budget:         ## E3 utility-fidelity-privacy frontier
	tilt run budget --config $(CONFIG) --resume
run-sampler:        ## E4 SIR vs twisted SMC at equal compute
	tilt run sampler --config $(CONFIG) --resume
run-validity:       ## E5 validity floor / DCR gate on-off
	tilt run validity --config $(CONFIG) --resume
run-scale:          ## E6 Qwen ladder 0.6B..32B (H100)
	tilt run scale --config h100_scale --resume
run-scarcity:       ## E7 q in {.05,.1,.2,.5}
	tilt run scarcity --config $(CONFIG) --resume
run-replication:    ## E8 thesis_main on the ImbLLM tables, q = 0.2
	tilt run thesis_main --config replication --resume

# ---------------------------------------------------------------- reports
.PHONY: report
report:         ## EXP=<experiment> METRIC=gain_vs_threshold|f1_tuned|mia_auc|...
	tilt report $(EXP) --metric $(METRIC) --figure

.PHONY: clean-models
clean-models:   ## delete cached fine-tuned models
	tilt clean --models

.PHONY: help
help:
	@echo "palt4 / tiltgen -- TILT: task-informed LLM tilting for imbalanced tabular oversampling"
	@echo "vars: DATASETS=$(DATASETS) SEEDS='$(SEEDS)' MODEL=$(MODEL) HW=$(HW) CONFIG=$(CONFIG)"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n",$$1,$$2}'
