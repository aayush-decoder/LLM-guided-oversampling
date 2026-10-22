# Thesis plan

**Working title.** *TILT: Task-Informed LLM Tilting for Imbalanced Tabular Classification*

**Target.** B.Tech thesis, plus one paper. Options:
- a workshop, or the main track of an ML or data-mining venue;
- aim for a venue whose deadline falls 2–3 weeks after E1 finishes;
- check each venue's call for papers for dates. None are assumed here.

## Abstract (states the hypotheses, which are not yet results)

LLM oversamplers for imbalanced tables (GReaT, Pred-LLM, ImbLLM) try to reproduce the minority distribution. We first show a limit that is simple but under-appreciated: at population level, a perfect minority generator only moves the decision threshold. No oversampler, tilted or not, beats a tuned threshold. Any real gain must therefore come from finite samples, misspecification, or the LLM's prior.

We then ask *which* synthetic rows reduce finite-sample error. We derive that the KL-budgeted first-order optimum is an exponential tilt of the generator by a Fisher-leverage utility: rows that are both near the boundary and in data-starved regions.

**TILT** samples this tilted distribution from a fine-tuned LLM row writer. It uses either whole-row importance resampling, or field-by-field twisted sequential Monte Carlo with a partial-row twist. A validity floor rules out label-noise rows. A record-amplification cap and a membership attack bound the privacy cost.

We evaluate on 10 naturally imbalanced tables and the ImbLLM benchmark:
- 4 downstream learners and 10 seeds;
- a single headline metric, the F1 gain over a tuned threshold;
- a no-LLM oracle study that isolates the sampling principle.

## Research questions

| RQ | question | experiment |
|---|---|---|
| RQ1 | When a generator is perfect, can utility tilting still help, and in which regime (n_min, dimension, imbalance ratio)? | E0 (oracle) |
| RQ2 | Does TILT beat the tuned threshold, the SMOTE family, ImbLLM, v2 and CLLM on real tables? | E1 |
| RQ3 | Which utility? Is variance (fisher / epistemic) better than ambiguity (aleatoric)? | E2, E5 |
| RQ4 | What does the budget trade: utility vs fidelity vs privacy? | E3 |
| RQ5 | Is field-wise twisted SMC worth it over SIR at equal compute? Does backbone scale change the picture? | E4, E6 |

## Contributions (what will be claimed, with the honest status)

1. **C1, theory.**
   - Prop. 1: the population no-gain result.
   - Prop. 2: first-order optimal placement is an exponential tilt, with influence and Fisher-leverage utilities.
   - Prop. 3: exactness of field-wise twisted SMC, with error bounds.
   - Prop. 4: privacy amplification.
   - The corollary that sequence-level CFG is a tilt.

   All are checked numerically (`tilt audit theory`). Novelty: NEW assembly and propositions on KNOWN foundations (THEORY.md).
2. **C2, method.** TILT: the KL-budgeted tilt, the validity floor, the amplification cap, and SIR/SMC samplers for LLM row writers. Novelty: the combination is new; the closest prior work is diffusion guidance for images (RELATED_WORK.md).
3. **C3, evaluation protocol.** `gain_vs_threshold` as the headline; 4 learners; the natural-imbalance benchmark; the membership attack on minority rows; the oracle study.
4. **C4, empirical findings.** To be produced. The thesis is publishable whichever way H2/H3 go:
   - a positive result is a method paper;
   - a negative result is a clear-eyed paper on where LLM oversampling can and cannot help, backed by theory.

## Chapters

1. **Introduction.** The imbalance problem; LLM oversamplers; the threshold question.
2. **Background and related work.** Cost-sensitive learning; the SMOTE family; deep and diffusion generators; LLM tabular generation; guided decoding; SMC; privacy of synthetic data (RELATED_WORK.md).
3. **Theory.** Props. 1–4 with proofs (THEORY.md).
4. **Method.** TILT (METHOD.md); implementation (MODULES.md).
5. **Experimental setup.** Datasets, protocol, learners, metrics, statistics, compute (EXPERIMENTS.md, DATASETS.md, COMPUTE.md).
6. **Results.** E0 oracle → E1 main → E2–E5 ablations and frontier → E6 scale → E7 scarcity → E8 replication → E9 contamination.
7. **Discussion.** When to use an LLM oversampler at all; limits (first-order theory, validity-probe quality, cost); privacy.
8. **Conclusion and future work.** DP fine-tuning; text-rich tables (palt/docs/FEASIBILITY_FR_EN.md); order-consistent training (palt3 order study).

## Timeline (relative; ~20 H100 hours per week)

| weeks | milestone | done when |
|---|---|---|
| 1 | Setup, smoke, **E0 oracle (go/no-go)** | `tilt smoke` passes; the oracle summary shows the regime where tilting helps (or clearly does not) |
| 2–4 | E1 on the natural set | `tilt report thesis_main --metric gain_vs_threshold` for all 10 tables |
| 5 | E2, E3, E5 (sampling-only, cheap) | frontier figure |
| 6 | E4, E6 | cost and scale figures |
| 7 | E7, E8, E9 | all tables filled |
| 8–10 | Writing: paper draft, then thesis chapters | – |
| 11–12 | Buffer, reruns, reviewer-style audit of claims | every claim traced to a `runs/` manifest |

## Risks and fallbacks

| risk | sign | fallback |
|---|---|---|
| Tilting does not beat untilted oversampling (H2 fails) | E0 after 30 seeds | Map the regime (IR, n_min, d); report it. The theory still stands. Tighten the claim to "when and why". |
| TILT beats v2 but not `threshold_only` (H3 fails) | E1 | The central finding becomes "LLM oversampling is not worth it for GBDT, even when guided". It is valuable and consistent with Prop. 1. Show which learners still gain (LR, MLP). |
| The validity probe is weak on some tables | `tilt_valid_share` near 1 on majority-like pools | Raise `valid_q`, or switch the probe learner. Report per table. |
| H100 hours run short | – | Qwen3-1.7B only; 5 seeds; skip the 14B/32B rungs (the scale axis becomes future work). |
| Memorisation of public tables | E9 flags adult / bank | Lead with the natural set (less-known tables); report the flags. |

## Verified so far (28 Sep 2026)

- The code builds; 104 fast tests pass.
- All six theory checks pass (`tilt audit theory`).
- One 3-seed oracle trial ran (EXPERIMENTS.md, E0 first look).
- **Not yet done:** no LLM training has run, no real-table results exist, and the torch-based tests (`pytest -m llm`, `tilt smoke`) have not run.
