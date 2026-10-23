# v3: Negative Fine-Tuning & Outlier Support LLM Tabular Generation

v3 implements negative majority unlikelihood training, outlier support bounding, and anomaly score gating on top of Pred-LLM and schema-constrained decoding.

See detailed theoretical derivation and ablation designs in [v3/docs/NOVELTY.md](docs/NOVELTY.md).

## Quick Start

```bash
# Run unit tests
python -m pytest v3/tests/test_v3_units.py -v

# Run smoke test end-to-end (iris, tiny-gpt2)
python -m predllm_v3 smoke

# Run single dataset with default v3 method
python -m predllm_v3 run --dataset iris --method pred_llm_v3

# Run ablations across presets
python -m predllm_v3 ablation --datasets iris

# Compare results across all versions
python -m tabgen report --versions v1,v2,v3
```

## Structure

```
v3/
├── configs/
│   └── default.yaml         # Config with negative_weight, outlier bounds, anomaly gating
├── docs/
│   └── NOVELTY.md           # Mathematical formulation and ablation rationale
├── predllm_v3/
│   ├── codec.py             # Schema with outlier ranges and minority/majority partitions
│   ├── outlier.py           # IQR / quantile bounds + IsolationForest anomaly scorer
│   ├── finetune.py          # V3RowDataset with majority unlikelihood penalty
│   ├── decoding.py          # Numeric FSM + CFG negative logit guidance
│   ├── sampler.py           # Field-wise sampling bounded to outlier envelopes
│   ├── posterior.py         # Permutation-marginalised exact label posterior
│   ├── curation.py          # DCR privacy gate + Anomaly filter + SIR
│   ├── generator.py         # PredLLMV3 pipeline
│   ├── methods.py           # Presets and ablations
│   ├── smoke.py             # Smoke runner
│   └── cli.py               # v3 command line interface
└── tests/
    ├── test_v3_units.py     # Unit tests for outlier bounds, loss, and gating
    └── test_v3_smoke.py     # End-to-end smoke test
```
