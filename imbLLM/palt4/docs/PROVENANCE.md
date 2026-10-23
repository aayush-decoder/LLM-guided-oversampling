# Provenance: what palt4 took from palt3

On 28 Sep 2026, `palt3/src/imbllm`, `palt3/configs/hardware` and `palt3/tests` were copied into palt4. palt3 was not modified.

- palt3 was never committed, so there is no commit hash.
- Its state is the one described in `palt3/docs/RESEARCH_REVIEW.md` §7 and `what_corrected.md` §H.
- At that point it had 83 fast tests passing.

## Mechanical changes

**Renames:**
- Package `imbllm` → `tiltgen`.
- CLI `imb` → `tilt`.
- Environment variables `IMBLLM_*` → `TILT_*`: `TILT_DATA`, `TILT_RUNS`, `TILT_CACHE`, `TILT_LEARNERS`.

**Kept:** ImbLLM *method* names (`imbllm_paper`, `full_imbllm`, `imbllm_minus_inter`), because they name the paper, not the package.

**Configs:**
- `laptop_qwen`, `h100_scale` (palt3) and `fast` were replaced by `laptop_dev`, `laptop_main`, `h100_main`, `h100_scale` (TILT) and `replication`.
- `paper` and `smoke` were kept.

## Changes to vendored code

| file | change |
|---|---|
| `core/data/fetch.py` | OpenML default-target option |
| `core/data/registry.py` | the `natural` benchmark |
| `core/llm/zoo.py` | Qwen 14B/32B rungs; Qwen3.5 and MoE rejected |
| `core/baselines/classical.py` | Borderline-SMOTE, SVM-SMOTE |
| `experiments/{grids,report,runner}.py` | TILT grids, `gain_vs_threshold`, kind `tilt`, membership attack |
| `cli.py` | `oracle`, `audit theory`, `--method tilt` |

Everything else in `methods/` apart from `methods/tilt`, and everything in `audits/` apart from `theory_checks.py`, is palt3's code, whose docstrings record its own corrections.

## Where the palt3 reasoning lives

These files were deliberately not copied, so there is one source of truth for each:
- `palt3/docs/what_corrected.md`: every IMPL / PROTO / METHOD fix of the ImbLLM re-implementation.
- `palt3/docs/PROTOCOL.md`: the 15 resolved ambiguities of the ImbLLM paper.
- `palt3/docs/RESEARCH_REVIEW.md`: the review that led to palt4's direction (§8, D1: "why should an LLM help at all?").
