# v2: Pred-LLM with schema-constrained decoding and calibrated labels

Same evaluation protocol as v1, so the numbers compare directly. Five changes (N1-N5) are explained
with the math in [docs/NOVELTY.md](docs/NOVELTY.md).

## Modules (`predllm_v2/`)

| module | novelty | what it does | LLM? |
|---|---|---|---|
| `codec.py` | N1, N3 | schema (real categories/units, per-column precision, ranges); canonical piece-wise tokenization; per-token loss weights | |
| `finetune.py` | N3 | D'_real (permute_x copy + original copy) with weights for the shared weighted loss | |
| `decoding.py` | N2 | exact candidate scoring with KV-cache reuse; numeric FSM-masked sampling | **yes** |
| `sampler.py` | N2 | field-wise feature-conditional sampling of x; the decoder writes keys and the LLM writes values; 100% valid | **yes** |
| `posterior.py` | N4 | p̂(y\|x) averaged over R feature orders (exact class scoring); regression by constrained draws | **yes** |
| `curation.py` | N5 | DCR privacy gate + cross-fitted discriminator SIR on a γ·n pool | |
| `methods.py` | | presets: `pred_llm_v2` and ablations `v2_no_curation`, `v2_R1`, `v2_argmax`, `v2_plain_loss`, `v2_encoded` | |
| `generator.py` | | wires everything, model cache, maps output to the shared encoded space | **yes (fine-tuning via `tabgen.llm`)** |
| `cli.py`, `smoke.py` | | same commands as v1 | |

## Quick start
```bash
python -m predllm_v2 smoke
python -m predllm_v2 run --dataset adult --plots
python -m predllm_v2 ablation --datasets classification
python -m tabgen report --versions v1,v2
```

The config is `configs/default.yaml`. Every key can be overridden, e.g.
`--set query.permutations=1 curation.enabled=false`.
