# v2: what changes relative to Pred-LLM, and why

v1 reproduces Pred-LLM exactly: text serialization of MinMax-encoded rows, `permute_x` fine-tuning,
feature-conditional **free-text** sampling, then querying the label with a prompt. v2 keeps the paper's
three good ideas (target-last permutation, feature-conditional start, labels from p(y|x)) and replaces
the parts that waste the LLM or add noise. Each change can be switched off on its own
(`python -m predllm_v2 ablation`).

Notation: a row is (x, y) with features x = (x_1..x_M). The serialized row is a token sequence c_1..c_L,
π is a feature order, and θ are the LLM parameters.

---

## N1. Semantic serialization (architecture / data)

**Paper (v1).** Following the authors' `read_data.py`, categories become integer codes and then every
column is MinMax-scaled. The LLM therefore reads `workclass is 0.43`, `occupation is 0.79`.
This throws away the main reason to use a pretrained LM for tables: its prior knowledge of what
`Bachelors`, `Exec-managerial` or `age is 39` mean. It also turns categories into fake numbers, so
free decoding can produce `0.44`, which is not a valid category.

**v2.** Serialize the *clean* table: real category strings and real units, each numeric column
rounded to its own inferred precision (`decimals_c` = the smallest d ≤ 4 that represents 99% of
values exactly). After generation, rows are mapped through the **same** fitted `TablePreprocessor`
as the real data, so TSTR is scored in exactly the same feature space as v1 and the baselines.

Ablation: `v2_encoded` (`serialization=encoded`).

---

## N2. Schema-constrained, field-wise decoding (architecture + math)

**Paper.** Rows are decoded as free text with token-level temperature T, then parsed; rows that are
malformed are thrown away and re-generated (rejection sampling at the row level).

Two issues:

1. *Row-level rejection biases the distribution.* The accepted rows follow
   p(x | row is well-formed) rather than p(x). Rows whose decoding is "hard" (long values, rare
   categories) are rejected more often, so they are under-represented in D_fake.
2. *Token-level temperature distorts multi-token values.* For a value v = (t_1..t_k),
   Π_i softmax(z_i/T) is not proportional to p(v)^{1/T}. Values that span more tokens are
   flattened differently from single-token values.

**v2.** The decoder writes every key (`" age is"`) and separator itself. The LLM supplies only values.

* **Exact categorical sampling.** For column j with categories V_j and prefix h:

  s(v) = log p_θ(tok(v), SEP | h) = Σ_{i=1..k+1} log p_θ(c_i | h, c_<i)

  p_T(v | h) = exp(s(v)/T_cat) / Σ_{v'∈V_j} exp(s(v')/T_cat)

  This is the model's own distribution restricted to the valid support and renormalized there,
  with a *sequence-level* temperature. Including the terminator token (SEP, or EOS for the target)
  in s(v) matters: without it, a value that is a token-prefix of another value (`Married` vs
  `Married-civ-spouse`) would be over-scored. It is computed in one prefix pass plus batched candidate
  passes that reuse the prefix KV cache (`decoding.score_continuations`, checked against a
  full-recompute reference in the tests).

* **Grammar-masked numeric sampling.** Digits are sampled token by token at temperature T_num, with
  logits masked by a finite-state automaton A_j for the column:

  `␠{0,2} -? d{1..I_j} ( . d{1..P_j} )? ⟨SEP⟩`

  where I_j is the number of integer digits of max|x_j| and P_j = decimals_j, so

  p̃(c_t | h) ∝ exp(z_t / T_num) · 1[c_t ∈ A_j(state_t)]

  The result is clipped to the training range. Invalid numbers are impossible by construction,
  so validity = 100% and nothing is rejected.

* **Canonical piece-wise tokenization.** Keys, values and separators are tokenized separately and
  concatenated, both in training and in decoding. The token sequence of a value therefore never
  depends on its neighbours. That is what makes exact scoring well defined and lets the decoder
  write scaffold tokens that are exactly in-distribution.

The feature-conditional start is kept: the first column X_c is cycled uniformly (n/M rows each)
with x_c ~ empirical p(X_c), and the remaining features follow a random order, which matches the
`permute_x` training distribution.

---

## N3. Weighted value-only objective (math)

**Paper.** Plain NLL over every token (Eq. 1). About half of those tokens are deterministic scaffold
(`age is`, `,`) that the model must still spend capacity on, and the label is 1 of ~50 tokens.

**v2.**

L(θ) = − E_{(x,y), π} [ Σ_{t ∈ K} w_s log p_θ(c_t|c_<t) + Σ_{j} Σ_{t ∈ V_j ∪ {term_j}} log p_θ(c_t|c_<t) + λ Σ_{t ∈ V_y ∪ {EOS}} log p_θ(c_t|c_<t) ] / Σ_t w_t

* K are key tokens, weighted w_s = 0 by default (the decoder writes them).
* V_j are value tokens of feature j, plus the terminator right after the value. The terminator
  carries the "number is finished" decision the numeric FSM relies on, so it must stay in the loss.
* λ > 1 (default 3) up-weights the target tokens, which directly trains the conditional p(y | x)
  that TSTR measures.

The implementation is `tabgen.llm.train_loop.weighted_lm_loss`, fed per-token weights by `codec.encode`.
Ablation: `v2_plain_loss` (λ = 1, w_s = 1, i.e. the paper's loss).

---

## N4. Permutation-marginalized exact label posterior (math)

**Paper.** Prompt with the features in original order and *sample* y from the free-text continuation
at T = 0.7. This has two sources of noise: the chosen order, and tokenization of the label.

**v2.** An autoregressive LM is not exchangeable: p_θ(y | π(x)) depends on π, while the true
posterior does not. We average over R orders (π_1 = original, the rest random):

p̂(y = c | x) = (1/R) Σ_r  exp s_r(c) / Σ_{c'} exp s_r(c'),   s_r(c) = log p_θ(tok(c), EOS | π_r(x), "y is")

Every p_r is exact (all classes are scored), and averaging over orders is a Monte-Carlo estimate of the
order-marginal predictor E_π[p_θ(y | π(x))], which is permutation-invariant. As a mixture of R
estimates its variance shrinks roughly as 1/R. Labels are then drawn as y ~ p̂^{1/T_y} (default
T_y = 0.7, matching the paper's sharpening) or set to argmax p̂ (`v2_argmax`).

For regression, y_r is sampled with the numeric FSM after each order and y = mean_r y_r, a Monte-Carlo
posterior mean that is MSE-optimal under the model (`reg_aggregate=single` gives the paper-like
single draw).

Ablation: `v2_R1` (R = 1).

---

## N5. Discriminator-guided resampling + DCR privacy gate (math)

Generate a pool of γ·n rows (default γ = 1.5), then keep n.

1. **Privacy gate.** Split D_real into a reference set R (90%) and a holdout H (10%), and set
   τ = Quantile_α{ DCR(h, R) : h ∈ H } with α = 0.01. A synthetic row with DCR(x̂, R) < τ is closer to a
   training record than 99% of *genuinely new* real rows are, so it is dropped as a near-copy.
   There are two safeguards: if τ = 0 (purely discrete data, where exact matches are unavoidable)
   nothing is dropped, and at most `max_drop` of the pool is ever removed.

2. **Sampling-importance-resampling with a density-ratio estimate.** A K-fold cross-fitted XGBoost
   D(x, y) separates real (1) from pool (0). With class priors corrected,

   r(x) = p_real(x) / p_syn(x) ≈ (n_syn / n_real) · D(x) / (1 − D(x))

   The weights are w_i = clip(r(x_i), 1/c, c)^β, and n rows are drawn without replacement with
   P(i) ∝ w_i. With β = 1 and a perfect D this is SIR, whose output approaches p_real as γ grows.
   β < 1 tempers the correction. The effective sample size (Σw)² / Σw² is logged. Cross-fitting
   keeps D from scoring rows it was trained on.

This directly targets the paper's own fidelity metrics (discriminator score ↓, inverse KL ↑,
density ↑) without touching the test set: τ uses an internal split of D_real only.

**Evaluation caveat (fixed 28 Sep 2026).** Because curation selects rows with an XGBoost
real-vs-synthetic discriminator fitted on the *training* rows, an XGBoost discriminator on the
training rows would grade v2 with its own filter, and the DCR gate would pass its own DCR test.
`tabgen/eval/fidelity.py` therefore measures fidelity (discriminator, `discriminator_lr`,
inverse KL, density, coverage) against the **held-out test rows**. Privacy (DCR, exact copies)
is measured against the training rows, and `discriminator_train` is kept for comparison.
Always report `v2_no_curation` next to `pred_llm_v2`.

Ablation: `v2_no_curation`.

---

## Summary

| Stage | Pred-LLM (v1) | v2 | Code |
|---|---|---|---|
| Serialization | MinMax codes, 2 decimals | real categories and units, per-column precision | `codec.build_schema` |
| Tokenization | whole sentence | canonical per-piece, cached | `codec.FieldCodec` |
| Loss | NLL on all tokens | value-only, λ on target | `codec.encode`, `train_loop` |
| Categorical decoding | free text, token-level T | exact enumeration, sequence-level T | `decoding.score_continuations` |
| Numeric decoding | free text + parse + reject | FSM-masked logits, clipped | `decoding.decode_numeric` |
| Validity | rejection sampling | 100% by construction | `sampler` |
| Label | 1 sampled completion, original order | exact posterior averaged over R orders | `posterior` |
| Post-processing | clip to [-10, 10] | DCR gate + discriminator SIR | `curation` |
| LLM scale | distilgpt2, full FT | any registry LLM (full / LoRA / QLoRA) | `tabgen.llm` |

## Prior art: what is new and what is not

A literature check on 28 Sep 2026 (details and ids in `palt3/docs/RESEARCH_REVIEW.md` §4)
found each component published in some form. Present them as **applications of known ideas to
GReaT-style generators**, not as inventions.

| | closest prior work | status |
|---|---|---|
| N1 semantic serialization | GReaT and TabuLa serialise raw values; Pred-LLM's MinMax codes are the outlier | known |
| N2 constrained decoding | REaLTabFormer masks tokens per column; Outlines / XGrammar do this in general | partly known (exact categorical enumeration + numeric FSM in an "X is v" generator is the delta) |
| N3 value-only weighted loss | TabularMDLM (noises value tokens only), prompt-loss masking in instruction tuning | close to known |
| N4 order-marginalised label posterior | permutation self-consistency (Tang et al. 2024), TabPFN permutation ensembling | known idea, new application |
| N5 discriminator SIR + DCR gate | discriminator rejection sampling (Azadi et al. 2019), Grover et al. 2019, CLLM curation (Seedat et al. 2024) | known; compare against CLLM |

A method paper built on v2 therefore needs a clear win over TabSyn / TabDDPM / CLLM (see the
review), not only over the 2024 Pred-LLM baselines.

## Cost

* Categorical scoring costs one prefix pass plus ⌈K / chunk⌉ short candidate passes per field
  (K = number of categories). This is cheaper than free decoding plus rejection when validity is
  low, and more expensive for columns with hundreds of categories (for example fuel `MODEL`).
* Label querying costs R× the paper's query. R = 4 is the default; use `--set query.permutations=1`
  on slow hardware.
* Curation generates γ× more rows plus a K-fold XGBoost.

## How to verify each claim

```bash
python -m predllm_v2 ablation --datasets classification --llm distilgpt2
python -m tabgen report --versions v1,v2
```

`reports/tables/table_accuracy.md` then shows v1 `pred_llm` (the paper), `pred_llm_v2` and each
`v2_*` ablation side by side with the paper's numbers. `fidelity.md` has the Fig. 5 metrics.
