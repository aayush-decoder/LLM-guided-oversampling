# TILT: the method

**Task-Informed LLM Tilting.** Generate the minority rows the classifier *needs*, not just rows that look like the minority. The generator stays within a stated fidelity budget and a privacy cap.

## 1. Target

\[
q^\star(x) \;\propto\; p_{\text{gen}}(x)\;\exp\!\big(\lambda\,z(x)\big)\;\mathbb 1[x\ \text{label-valid}]\;\mathbb 1[x\ \text{passes the DCR gate}]
\]

| symbol | meaning | where in the code |
|---|---|---|
| \(p_{\text{gen}}\) | the fine-tuned LLM row writer: v2_corrected `recommended`, with the same corpus and the same cached fine-tune | `methods/v2_corrected`, reused by `methods/tilt/method.py` |
| \(z\) | standardised utility, \((u-\bar u)/s_u\) clipped to ±3 | `tilt.standardise` |
| \(u\) | utility of a row to the downstream learner | `methods/tilt/utility.py` |
| \(\lambda\) | the largest value with \(\mathrm{KL}(q\|p_{\text{gen}})\le\varepsilon\), capped by amplification \(A\) | `tilt.calibrate_lambda` |
| validity | the class-balanced non-linear probe must rate the row at least as minority-like as the 5% least-typical real minority rows (out of fold) | `UtilityModel.valid` |
| DCR gate | drop rows closer to a real minority training row than the 5% quantile of genuinely new distances | `methods/tilt/privacy.py` |

Why this form: it is the exact solution of "maximise expected utility subject to staying within ε nats of the generator" (THEORY.md, Lemma 1 and Prop. 2).

## 2. Utilities

All utilities are fitted on the real imbalanced training set only, with 2-fold cross-fitting.

| name | formula | reading | role |
|---|---|---|---|
| `fisher` | \(\sigma(1-\sigma)\,\tilde x^\top H^{-1}H_{\text{val}}H^{-1}\tilde x\) | boundary × data-starved; label-free variance reduction (Prop. 2) | **default** |
| `influence` | \((1-\sigma)\,g^\top H^{-1}\tilde x\), with \(g=-\nabla R_{\text{val}}\) | first-order drop of the balanced validation loss if x is added as minority | ablation |
| `epistemic` | std of logit over 8 bootstrap probes | where more data would change the model | ablation, model-agnostic |
| `aleatoric` | \(4p(1-p)\), from the non-linear balanced probe | ambiguous rows | ablation, **predicted to hurt** |
| `none` | 0 | λ = 0, so TILT is the untilted generator | control |
| `contrast` | \(\log p_{\text{LM}}(x\mid y_{\text{maj}}) - \log p_{\text{LM}}(x\mid y_{\min})\), per token | sequence-level classifier-free guidance (THEORY.md corollary) | needs `majority_rows > 0` |

## 3. Samplers

### SIR (`tilt_sir`, default): whole rows

1. Generate a pool of `pool_factor · n` rows with the v2 generator (default ×4).
2. Keep only rows that pass the validity floor and the DCR gate.
3. Compute \(u\), then \(z\). Calibrate \(\lambda\) on the pool, using a self-normalised KL estimate:
   \(\widehat{\mathrm{KL}} = \sum_i \tilde w_i \log(N\tilde w_i)\), with \(\tilde w\propto e^{\lambda z}\).
4. Draw n rows **without replacement** \(\propto e^{\lambda z}\), by Gumbel-top-k. It never emits duplicates.

Cost: about `pool_factor` × plain generation. The fine-tune is shared with v2.

### Twisted SMC (`tilt_smc`): field by field

```
pilot  = v2 generator, 256 rows      -> utility scale, λ, twist training data
twist  = PartialUtility: gradient-boosted regressor of z on randomly masked rows
for each group of N particles (prompt "Y is y_min[, X_i is v]"):
    repeat until every particle has finished:
        extend each unfinished particle by ONE field (LLM continuation, cut at ", ")
        parse its complete clauses; invalid -> log ψ = -inf
        log ψ = λ·ẑ(x_S)                     (partial row)
              = λ·z(x) + log 1[valid]        (finished row)
        log w += log ψ_new - log ψ_old
        resample the group if ESS < N/2
    output `keep` particles ∝ final weights, without replacement
```

- Exact for any twist (THEORY.md, Prop. 3a).
- A good twist prunes bad partial rows early, instead of paying for whole rejected rows.
- Cost per output row is about (N/keep) × (M+1) short completions.

### Off (`tilt_off`)

λ = 0. This is v2 `recommended` exactly; the ablation anchor.

## 4. Knobs (`TiltConfig`)

| knob | default | meaning |
|---|---|---|
| `sampler` | `sir` | `off` \| `sir` \| `smc` |
| `utility` | `fisher` | see §2 |
| `kl_budget` | 0.25 nats | ε: fidelity budget (the `budget` experiment sweeps 0 to 2) |
| `amp_cap` | 20 | A: max record amplification \(e^{\lambda\,\mathrm{range}(z)}\) |
| `pool_factor` | 4 | SIR pool size / n |
| `pilot`, `particles`, `keep`, `ess_frac`, `field_tokens` | 256, 8, 4, 0.5, 24 | SMC settings |
| `valid_q` | 0.05 | validity floor quantile (0 disables it) |
| `privacy_alpha` | 0.05 | DCR gate quantile (0 disables it) |
| `majority_rows`, `contrast_sign` | 0, +1 | for `contrast` |
| everything in V2Config | `recommended` | the generator: kNN geometry, anchor categoricals, τ-bridge, snap_discrete, ordinal_interpolation, ... |

Every TILT knob acts at **sampling time**, so one cached fine-tune per (dataset, seed) serves every arm except `contrast`, whose corpus adds majority rows.

## 5. What is logged

**`diagnostics.json → tilt`:**
- λ, and which constraint bound it: KL budget or amplification;
- realised KL and ESS;
- pool size, eligible rows, validity share, DCR-gate drops;
- utility scale;
- mean z of the pool vs of the selected rows;
- for SMC: resamples, completions, rows that died invalid, final ESS.

**`metrics.json`** also carries:
- `tilt_lambda`, `tilt_kl`, `tilt_ess`;
- `mia_auc` / `mia_auc_kde`;
- `gain_vs_threshold`, which the report derives.
