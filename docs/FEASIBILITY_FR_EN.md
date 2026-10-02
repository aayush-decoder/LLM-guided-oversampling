# Feasibility report: English vs French data, French = 5% minority

**Question.** Can v1 (Pred-LLM) or v2 of this repo generate useful synthetic data for a table where
language ∈ {English, French} is the target, French is only 5% of rows, and the table includes text,
and how well would it perform?

**Method.** This is a static analysis of the current code plus arithmetic. Nothing was trained or run.
Figures marked *estimate* are expectations to be confirmed by the measurement plan in §6.

---

## 1. Verdict

| Your data looks like | v1 as-is | v2 as-is | After the changes in §5 |
|---|---|---|---|
| **A. A free-text column** (sentence, review, …) + language label | **Not feasible.** The text is reduced to an integer code, so no language signal reaches the LLM or the evaluator | **Not feasible.** It only replays training sentences, and cost grows with the number of training texts | **Feasible**, *estimate*: macro-F1 within a few points of the real-data baseline with a multilingual LLM |
| **B. No free text**: language is a label or feature next to ordinary numeric/categorical columns | Feasible, but needs the imbalance fixes (§4) | Feasible, but needs the imbalance fixes (§4) | Feasible |

The 5% imbalance alone is manageable. The blocker is **free text**: both versions treat any
non-numeric column as a *categorical* column with a finite set of values.

---

## 2. Why free text breaks the current pipeline (case A)

| # | Where | What happens to a text column | Effect |
|---|---|---|---|
| 1 | `data/preprocess.py` `TablePreprocessor` | every unique sentence becomes a category code, scaled to [0,1]: `text is 0.4312` | v1's LLM never sees the words; language is invisible |
| 2 | `eval/tstr.py` (all methods, v1 and v2) | XGBoost is trained on that code | the evaluator cannot read text either; codes are sorted alphabetically, so it only picks up weak "starts with *Le/La/Les*" clusters |
| 3 | `v2/codec.py` `build_schema` | the text column becomes `kind="cat"` with candidates = all unique training sentences | v2 can only **copy** an existing sentence; it never writes a new one |
| 4 | `v2/decoding.py` `score_continuations` | with ~30-token candidates and a 50k vocabulary, the memory cap gives 1 candidate per forward pass | N = 5,000 means ~4,000 candidates per batch of 64 rows. A 6,000-row pool (γ = 1.5) needs ~375k forward passes, i.e. hours to days |
| 5 | `v1/serialize.py` `parse` | generated values must parse as floats | v1 has no path for emitting text at all |
| 6 | `data/preprocess.py` `clean_table` | commas in text become `;` | harmless for language ID, but alters the text |

**LLM choice matters for French.** The default `distilgpt2` was pre-trained mostly on English. Its
byte-level tokenizer splits French words (accents, elisions) into more tokens than English words
(roughly 1.5–2×, *estimate*), so French rows are longer and its French priors are weak. Multilingual
base models in the registry (`qwen2.5-1.5b` with LoRA fits a 6 GB GPU, `qwen2.5-3b` with QLoRA)
are the right backbones.

---

## 3. How the 5% minority is handled (applies to A and B)

### 3.1 Evaluation statistics: the test set has very few French rows

The split is stratified 80/20, so French rows are 4% of N in train and 1% of N in test.

| total rows N | French train | French test | 95% CI half-width of French recall (true recall 0.9) |
|---|---|---|---|
| 1,000 | 40 | 10 | **±0.19** |
| 5,000 | 200 | 50 | ±0.08 |
| 20,000 | 800 | 200 | ±0.04 |

Below ~5,000 rows, differences between methods on the French class are mostly noise. Use more
seeds (`--runs 10`) or repeated splits.

### 3.2 Accuracy hides the problem

Predicting "English" for every row scores **0.95 accuracy**, with macro-F1 ≈ 0.49 and French recall 0.
The reports use accuracy as the headline (`eval/report.py`), and `tstr()` also returns `f1_macro`,
but there is no balanced accuracy, minority recall or PR-AUC.

### 3.3 Temperature 0.7 under-produces the minority

Both versions sharpen label sampling with T = 0.7 (v1 `sampling.temperature`, v2
`query.temperature_y`). Sampling from p^(1/T) and renormalizing shrinks small probabilities:

| p(French) before | after T = 0.7 |
|---|---|
| 0.05 | **0.015** |
| 0.30 | 0.23 |
| 0.50 | 0.50 |

When a row's language is uncertain, French is under-drawn. In v1, the same token-level sharpening
applies whenever the language is decided at a single token. The result can be ~1.5% French in D_fake
instead of 5%.
v2's categorical sampling uses T = 1.0 (`temperature_cat`), which does not have this problem.

### 3.4 Audits would not catch minority collapse

* `audit.py` class balance: `counts.min() >= 0.05` sits right at the boundary and passes or warns
  depending on rounding.
* `audit.py` label shift: `TV < 0.15`. If French falls from 5% to 1%, TV = 0.04, which **passes**.
  Only the complete absence of French is flagged (as a FAIL, and TSTR returns accuracy = 0 by the
  paper rule).

### 3.5 What already works in our favour

* **v2 label posterior** (`posterior.py`): each class is scored exactly and averaged over R feature
  orders. For language ID from real text the posterior is near 0 or 1, so labels will be clean.
* **v2 SIR curation** pulls the synthetic distribution toward the real one, including the 5% share.
* **Stratified splits** keep 5% French in both train and test.

---

## 4. Expected performance

| Setting | French recall | macro-F1 | accuracy |
|---|---|---|---|
| Real data (Original), text features via char n-gram TF-IDF | ~0.95–1.0 | ~0.97–1.0 | ~0.99+ |
| **A**, v1 or v2 as-is (text as code) | ~0–0.3 | ~0.5–0.6 | ~0.95 (misleading) |
| **A** after §5, v2 + `qwen2.5-1.5b` | ~0.85–0.98 | ~0.9–0.98 | ~0.99 |
| **A** after §5, v2 + `distilgpt2` | ~0.6–0.9 | ~0.8–0.93 | ~0.97–0.99 |
| **B** (no free text), v2 as-is | depends on how predictive the other columns are; expect minority recall a few points under Original, plus the T-sharpening loss of §3.3 unless T_y = 1 | | |

All numbers are *estimates*. English-vs-French identification on sentences is an easy task for
character n-grams. It gets hard mainly for very short texts (1–3 words), names, and code-switched
text. The uncertainty in the table comes from how well the fine-tuned LLM writes French from only
~4% of its training rows.

**Rule of thumb** (*estimate*): aim for at least ~200 French training rows (N ≳ 5,000) before
expecting fluent synthetic French from LoRA fine-tuning.

---

## 5. Changes needed to make case A feasible (ordered by impact)

1. **A `text` column kind (v2 codec/sampler).** Generate free text token by token, stopping at the
   field separator, instead of enumerating candidates. Escape `,` and ` is ` inside values.
   Keep exact scoring for real categoricals. *Moderate effort.*
2. **Text-aware evaluation.** For text columns, TSTR should use char n-gram TF-IDF + logistic
   regression (or XGBoost on TF-IDF) instead of alphabetic codes. Also add text fidelity checks:
   the share of synthetic "French" rows that a reference language identifier also calls French,
   and the n-gram overlap / exact-copy rate against training texts. *Moderate.*
3. **Imbalance-aware metrics and audits.** Report macro-F1, balanced accuracy, minority
   precision/recall and PR-AUC as the headline for imbalanced data. Replace the TV check with
   "synthetic minority share ≥ 0.5 × real share". *Small.*
4. **Minority-aware generation.**
   - Set `query.temperature_y=1.0` (no sharpening) or `label_decode=argmax`.
   - Add a class-targeted selection step to curation: generate a larger pool and keep a chosen
     French share, either 5% (faithful) or e.g. 30–50% (augmentation).
   - Optionally up-weight French rows in the loss. *Small.*
5. **Multilingual backbone.** Use `--llm 1.5b` (Qwen2.5-1.5B, LoRA) on a 6 GB GPU. *Configuration only.*
6. **More robust evaluation.** Use ≥ 10 seeds, or repeated stratified splits when N < 5,000. *Configuration only.*

Case B needs only items 3–4 and 6.

---

## 6. Measurement plan (once §5 is in place)

1. Add the dataset to `configs/datasets.yaml` with `task: classification`, the language column as
   `target`, and the text column listed separately (new `text:` key).
2. Get reference numbers: `python -m predllm_v2 run --dataset <name> --method original --runs 10`.
3. Compare: `python -m predllm_v2 run --dataset <name> --methods pred_llm_v2,v2_argmax --llm 1.5b --runs 10 --plots`,
   plus v1 `pred_llm` and the SDV baselines.
4. Accept if French recall on the real test set is within 5 points of Original, macro-F1 is within
   2 points, the synthetic French share is between 4% and 6% (faithful mode), and the exact-copy
   rate is below 5%.
