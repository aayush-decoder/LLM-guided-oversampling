# Theory

This file states what TILT rests on and marks each statement's status:
- **KNOWN** — published; cite it.
- **NEW** — ours to prove; the proof sketches are here.
- **CHECKED** — verified numerically by `tilt audit theory` (`src/tiltgen/audits/theory_checks.py`).

Citation tags:
- **[V]** — the statement was read in the source during the 28 Sep 2026 literature pass.
- **[M]** — from memory. Check the theorem number in the PDF before citing it.

Throughout this file:
- \(y\in\{0,1\}\); 1 is the minority.
- \(\pi = n_{\min}/(n_{\min}+n_{\text{maj}})\).
- \(\eta(x)=P(y=1\mid x)\).
- \(p_{\min}=p(x\mid y=1)\).
- A generator adds \(m\) synthetic minority rows drawn from \(q\).

---

## 1. The baseline every oversampler must beat

### Proposition 1 — at population level, oversampling is a threshold move (KNOWN pieces, NEW assembly)

Add \(m\) rows from \(q \ll p_{\min}\), and let \(r = q/p_{\min}\). The augmented population then has posterior odds

\[
\frac{\eta'(x)}{1-\eta'(x)} \;=\; \frac{n_{\min}\,p_{\min}(x) + m\,q(x)}{n_{\text{maj}}\,p_{\text{maj}}(x)}
\;=\; \Big(1 + \tfrac{m}{n_{\min}}\, r(x)\Big)\,\frac{\eta(x)}{1-\eta(x)} .
\]

So the augmented Bayes rule is \(h_q(x) = \mathbb 1\big[\tfrac{\eta}{1-\eta} > \tfrac{n_{\min}}{n_{\min} + m\,r(x)}\big]\): **a threshold on \(\eta\) that may depend on \(x\).**

**(a) Perfect oversampling.** With \(q=p_{\min}\) (so \(r\equiv 1\)), \(h_q = \mathbb 1[\eta > t_m]\) with \(t_m = n_{\min}/(2n_{\min}+m)\). That is a *constant* threshold.
- This is Elkan (2001, IJCAI) Thm 1 [V] and the base-rate theorem, Thm 2 [V exists; formula M].
- For well-specified logistic regression, only the intercept moves, by \(\log\frac{n_{\min}+m}{n_{\min}}\) (King & Zeng 2001 [M]).

**(b) Nothing beats the best threshold.** For linear-fractional metrics (minority F1, balanced error, Jaccard), the Bayes-optimal classifier is a thresholded \(\eta\) (Koyejo et al. 2014, NeurIPS [V]; for F1, Lipton et al. 2014 [V]; for balanced error, Menon et al. 2021 [M]). Hence, for every \(q\):
\[ \mathrm{M}(h_q) \;\le\; \sup_t \mathrm{M}(\mathbb 1[\eta>t]) , \]
with equality iff \(h_q\) agrees almost everywhere with some global threshold. **Tilting (\(r\not\equiv 1\)) can never beat the tuned threshold at population level.**

**Consequence (the thesis's framing).** Assume the learner is consistent for the augmented population (A2) and that \(q\) contains no information outside the training data. Then any real gain of an oversampler over `threshold_only` must come from one of:
1. **Finite samples:** variance, i.e. regularisation of an inefficient learner.
2. **Misspecification:** reweighting changes the best-in-class *slope*, not only the intercept.
3. **\(q \ne p_{\min}\):** a deliberate bias, which TILT chooses.
4. **Outside information:** the LLM's pre-training prior.

This is why `threshold_only` (tuned on a real validation fold) is the anchor of `thesis_main`, and why the headline metric is `gain_vs_threshold`.

**CHECKED** by `prop1_population`, a 2-D Gaussian with 3,000 minority and 60,000 majority rows:
- F1 of the tuned threshold is 0.3656; F1 of perfect oversampling plus a tuned threshold is 0.3660.
- The intercept shift is 2.96, against the theory's 2.996.

Relation to Nakada et al. (arXiv 2406.03628, v3 Feb 2026, Thm 1 [V]): with zero bias, synthetic oversampling at best recovers \(\theta_{\text{bal}}\), the reweighting solution. Their analysis treats every bias \(\mathcal B\ne 0\) as harmful. TILT's claim is that a *chosen* bias, bounded by a KL budget, can help in finite samples.

---

## 2. Where synthetic rows should go

### Lemma 1 — Gibbs variational principle (KNOWN)

Donsker & Varadhan (1975); Dupuis & Ellis (1997) Prop 1.4.2 [M]; Catoni (2007) Lemma 1.1.3 [M]:
\[
\max_q\; \mathbb E_q[u] - \tfrac1\lambda \mathrm{KL}(q\,\|\,p) \;=\; \tfrac1\lambda \log \mathbb E_p e^{\lambda u},
\qquad q^\star(x) = \frac{p(x)\,e^{\lambda u(x)}}{Z_\lambda}.
\]

*Proof.* \(\mathbb E_q u - \frac1\lambda\mathrm{KL}(q\|p) = \frac1\lambda[\log Z_\lambda - \mathrm{KL}(q\|q^\star)]\).

- The constrained form, \(\max \mathbb E_q u\) subject to \(\mathrm{KL}(q\|p)\le\varepsilon\), has the same solution, with \(\lambda\) as the Lagrange multiplier.
- For small \(\varepsilon\), the gain is \(\approx \sqrt{2\varepsilon\,\mathrm{Var}_p(u)}\).
- **CHECKED:** in `lemma1_gibbs`, \(q^\star\) beats 2,000 random \(q\) on the objective.

### Proposition 2 — first-order optimal placement (NEW; the core result)

**Setting.** (Regularised) logistic regression at \(\hat\theta\), with Hessian \(H=\nabla^2\hat R(\hat\theta)\), on features \(\tilde x = [x, 1]\). Let \(R\) be a smooth validation objective; TILT uses the class-balanced log-loss on a held-out fold. Add synthetic rows with label 1 and total weight \(\varepsilon\), drawn from \(q\).

**The linearisation.** By the implicit function theorem, which is the Koh & Liang (2017) influence step:
\[
\Delta\theta = \varepsilon H^{-1}\,\mathbb E_q[(1-\sigma(\hat\theta^\top\tilde x))\,\tilde x] + O(\varepsilon^2),
\qquad
\Delta R = -\varepsilon\,\mathbb E_q[u_1(x)] + O(\varepsilon^2),
\]
\[
u_1(x) = (1-\sigma(\hat\theta^\top\tilde x))\; g^\top H^{-1}\tilde x, \qquad g = -\nabla R(\hat\theta). \tag{influence}
\]

**The optimum.** \(\Delta R\) is linear in \(q\). So by Lemma 1, the KL-budgeted first-order optimum is exactly the tilt \(q^\star\propto p_{\min}\,e^{\lambda u_1}\).

**Variance form.** Minimise the held-out variance proxy \(\operatorname{tr}\big(H_{\text{val}}(H+\varepsilon\,\mathbb E_q I(x))^{-1}\big)\), where \(I(x)=\sigma(1-\sigma)\tilde x\tilde x^\top\) does **not** depend on the label. The first-order gain is \(\varepsilon\,\mathbb E_q u_2\) with
\[
u_2(x) = \sigma(1-\sigma)\;\tilde x^\top H^{-1} H_{\text{val}} H^{-1}\tilde x . \tag{fisher, TILT's default}
\]
- \(\sigma(1-\sigma)\) peaks at the decision boundary.
- \(H^{-1}\)-leverage is large where training data are sparse.
- \(u_2\) is therefore the rigorous form of the Borderline-SMOTE intuition: *boundary, but where the model is data-starved*.

**Why variance and not ambiguity.** \(u_2\) is label-free and targets reducible, epistemic error. The aleatoric score \(4p(1-p)\) targets irreducible label noise. Tilting toward that noise with label 1 is the objection CLLM-style curation is built on (Seedat et al. 2024). TILT keeps it only as an ablation (`tilt_aleatoric`), predicted to hurt (H4 in EXPERIMENTS.md).

**Caveat: first order and local.** **CHECKED** by `prop2_first_order`: the predicted \(\Delta R\) is compared with a real refit across five tilts.

| synthetic mass ε | correlation (predicted vs refit) | slope |
|---|---|---|
| 0.002 | **0.997** | 0.90 |
| 0.05 (balancing scale) | 0.47 | 0.18 |

At the balancing scale the response saturates, and even reverses for the strongest tilt. This is the mathematical reason for a **KL budget** rather than "tilt as hard as possible": the linear law only holds near \(p_{\text{gen}}\).

---

## 3. Sampling q* correctly

### Proposition 3 — twisted SMC over fields is exact for any positive twist (KNOWN core, NEW application)

The generator writes a row field by field. Twisted SMC is Zhao, Brekelmans, Makhzani & Grosse (ICML 2024, arXiv 2404.17546); its Eq. 10 [V] gives the incremental weight. Sampling each field from the LLM itself makes the proposal equal to the model, so the weight simplifies to
\[
\log w_t \mathrel{+}= \log\psi_t(x_{1:t}) - \log\psi_{t-1}(x_{1:t-1}), \qquad \log\psi_T = \lambda z(x) + \log \mathbb 1[x\ \text{valid}] .
\]

(a) **Exactness.**
- The increments telescope to \(\lambda z(x)\).
- So the weighted particle system targets \(q^\star \propto p_{\text{LM}}\,e^{\lambda z}\mathbb 1[\text{valid}]\) for **any** positive intermediate twist \(\psi_{1:T-1}\).
- The twist only changes the weight variance and ESS [V: Eq. 8 and discussion; Del Moral 2004 M].
- The optimal twist is \(\psi_t^\star=\mathbb E[e^{\lambda z}\mid x_{1:t}]\), which gives zero-variance weights (Prop 3.2 [V]).
- TILT's twist \(\lambda\hat z(x_S)\) comes from a regressor trained on masked rows (`methods/tilt/partial.py`); this is FUDGE (Yang & Klein 2021) adapted to fields.
- **CHECKED:** in `prop3_smc_exact`, the shipped sampler reproduces \(q^\star\) on a two-field toy model: total-variation distance 0.021 from 5,000 draws (the pass threshold is 0.04), using both a flat and an informed twist.
- A pitfall found while building this check: if the toy model and the sampler share a random seed, their draws correlate and bias the result. The check keeps the seeds apart.

(b) **Utility error** (NEW, easy). If \(\|\hat u - u\|_\infty\le\delta\), then \(|\log q_{\hat u}/q_u|\le 2\lambda\delta\). Hence:
- \(\mathrm{KL}(q_{\hat u}\|q_u)\le 2\lambda\delta\);
- \(\mathrm{TV}\le\sqrt{\lambda\delta}\) (Pinsker).

(c) **Generator misfit** adds \(\mathrm{KL}(p_{\min}\|p_{\text{gen}})\). This is the \(\mathcal B\) term of Nakada et al. Thm 1.

(d) **Finite N** adds \(O(1/N)\) self-normalisation bias.

**SIR** is the one-step special case, with no intermediate twist. It is consistent as the pool grows.
- The pool must grow like \(\approx e^{\mathrm{KL}(q^\star\|p)}\) (Chatterjee & Diaconis 2018, Ann. Appl. Probab. [V]).
- For the tilt, \(\mathrm{KL}(q^\star\|p) = \lambda\mathbb E_{q^\star}u - \log Z_\lambda\).
- So the fidelity budget \(\varepsilon\) also bounds the compute: the budget does double duty.

### Corollary — contrast guidance is a tilt (NEW, one line)

With \(u(x) = \log p_{\text{LM}}(x\mid y_{\text{maj}}) - \log p_{\text{LM}}(x\mid y_{\min})\):
\[ p_{\min}\,e^{\lambda u} = p_{\min}^{1-\lambda}\,p_{\text{maj}}^{\lambda}, \]
which is classifier-free guidance (Sanchez et al. 2023, for LMs) at the **sequence** level. Token-level CFG does not sample this distribution exactly; SIR or SMC with this \(u\) does. TILT's utility `contrast` implements it.
- \(\lambda>0\) moves toward the majority, i.e. the boundary.
- `contrast_sign = -1` moves away from it.

---

## 4. Privacy

### Proposition 4 — tilting amplifies exactly the records attacks target (NEW, easy)

(i) **Exposure amplification.**
- Take a kernel-like generator \(\hat p = \frac1n\sum_i K_h(\cdot - x_i)\).
- The tilt gives record \(i\) a neighbourhood mass \(\approx w_i = e^{\lambda u(x_i)}/\sum_j e^{\lambda u(x_j)}\).
- Its amplification is therefore \(n w_i \le e^{\lambda\,\mathrm{range}(u)}\).
- Boundary records are amplified, and those are the records membership attacks hit hardest:
  - rare records leak (Feldman 2020 [M]);
  - outliers leak first (Carlini et al. 2022, the onion effect [V]);
  - outliers stay exposed in synthetic data (Stadler et al. 2022 [V]).
- TILT therefore caps \(\lambda \le \log A/\mathrm{range}(z)\). **CHECKED** by `prop4_amplification`: calibration never exceeds the cap.

(ii) **DP accounting.**
- If the generator is \((\varepsilon_1,\delta_1)\)-DP, post-processing covers the tilt **only if \(u\) is data-independent** (Dwork & Roth 2014, Prop 2.1 [M]).
- A \(u\) fitted on the data needs its own \((\varepsilon_2,\delta_2)\).
- Composition then gives \((\varepsilon_1+\varepsilon_2,\delta_1+\delta_2)\).
- Sampling \(\propto e^{\lambda u}\) is the exponential mechanism (McSherry & Talwar 2007).

**Practice.**
- The DCR gate is a guard, not a guarantee (Yao et al. 2025, "The DCR delusion" [V]).
- The reported privacy number is the membership attack in `core/evaluate/privacy.py`: a nearest-neighbour density ratio, DOMIAS-style (van Breugel et al. 2023 [M]).

---

## 5. What the thesis can claim, and what it cannot

| claim | status |
|---|---|
| Perfect oversampling ≡ a threshold move; nothing beats the tuned threshold at population level | KNOWN pieces, NEW assembly; CHECKED |
| The KL-budgeted first-order optimal placement is the exponential tilt by \(u_1\) or \(u_2\) | NEW (Prop 2); CHECKED for small ε; fails at balancing scale, which motivates the budget |
| Field-wise twisted SMC is exact for any positive twist; error bounds for estimated utilities | KNOWN core, NEW application and bounds; CHECKED on a toy model |
| Sequence-level CFG is a special case of the tilt | NEW (corollary) |
| Amplification bound; DP composition caveat | NEW (easy); CHECKED (cap) |
| "TILT beats `threshold_only` on real tables" | **UNTESTED.** This is the hypothesis of `thesis_main`. The oracle study (EXPERIMENTS.md, E0) is the cheapest first test. |
