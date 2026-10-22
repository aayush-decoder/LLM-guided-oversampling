# v1: Pred-LLM replication

A faithful re-implementation of *Generating Realistic Tabular Data with Large Language Models*
(Nguyen, Gupta, Do, Nguyen, Venkatesh; arXiv:2410.21717), checked against the authors' code
(github.com/nphdang/Pred-LLM).

## Modules (`predllm_v1/`)

| module | paper | what it does | LLM? |
|---|---|---|---|
| `methods.py` | Table III | `MethodSpec(permutation, sampling, query, augment_original)`, with presets `pred_llm`, `great`, `taptap` and 6 ablation rows | |
| `serialize.py` | III-B1a,b | `"X is v, ..., target is y"`, `permute_x` / `permute_xy`, parsing of generated text | |
| `finetune.py` | III-B1c, Alg. 1 l.2-12 | D'_real = permuted ∪ original sentences, re-shuffled on every fetch | |
| `sampler.py` | III-B2, Eq. 2 | feature-conditional (n/M rows per feature, start `X_i is v`) or target-conditional free-text sampling, T=0.7, top-k 50, parse + reject + retry | **yes (generation)** |
| `querier.py` | III-B3 | prompt `X_1 is v_1, ..., X_M is v_M, target is` → label, 5 retries; TapTap classifier; GReaT none | **yes (label prompting)** |
| `generator.py` | Alg. 1 | wires the phases together, handles the model cache and the `[-10, 10]` clip | **yes (fine-tuning via `tabgen.llm`)** |
| `cli.py` | Sec. IV | run / benchmark / ablation / train / generate / infer / evaluate / attention / smoke | |
| `smoke.py` | | end-to-end artifact check | yes |

Shared code (splits, encoding, XGBoost TSTR, metrics, SDV baselines, plots, audits, LLM switch and
training loop) lives in `common/tabgen`.

## Protocol (as in the paper and the authors' code)
* 80/20 stratified split per seed (seeds 0, 1, 2), with |D_fake| = |D_real|.
* Features become category codes and are MinMax-scaled, rounded to 2 decimals; labels become 0..K−1.
* distilgpt2, full fine-tuning, batch 32, 50 epochs, AdamW lr 5e-5 (HF Trainer default), linear decay.
* Sampling with T = 0.7, top-k 50 (HF default), k = 100 sequences per call.
* XGBoost with default parameters on D_fake, evaluated on D_test. If D_fake is missing a class,
  accuracy = 0 (paper rule).
* Fidelity metrics: discriminator (XGBoost, 2·AUC−1), inverse KL, density/coverage (k=2), DCR.

## Deliberate, documented deviations
* Each training sentence ends with EOS, and generation stops at EOS instead of a fixed `max_length`.
  This saves compute and gives the same result.
* The query prompt ends with `, target is` so the completion is always the label (`query.force_target_key`).
* Baselines: CTGAN, TVAE and CopulaGAN (SDV) are trained on the clean table (real categories), their
  native input. MedGAN, VeeGAN, TableGAN, TabGAN and OCTGAN are not included.
* TapTap is implemented as the paper describes it in Table III (permute_xy + target start +
  external classifier), not TapTap's own pre-trained table model.

## Outputs
`v1/runs/<dataset>/<method>__<llm>/seed<k>/` contains config, split, preprocessor, synthetic data
(encoded + clean), generation stats, metrics, train log and plots. Fine-tuned models are in
`v1/runs/<dataset>/_models/<llm>_<fingerprint>/` and are shared between presets with identical
fine-tuning. See `common/tabgen/experiment.py` for the full layout.
