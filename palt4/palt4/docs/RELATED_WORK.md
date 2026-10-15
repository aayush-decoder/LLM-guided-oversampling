# Related work and the novelty claim

This comes from a literature search on 28 Sep 2026 (arXiv, OpenReview, Semantic Scholar, GitHub). Treat arXiv ids as leads and check each before citing it.

## The defensible claim

We found **no** work that combines all four of these:

1. **An explicit objective.** LLM tabular oversampling cast as sampling from \(q^\star\propto p_{\text{gen}}\,e^{\lambda u}\) under an explicit fidelity budget \(\mathrm{KL}(q\|p)\le\varepsilon\). The objective includes a validity floor and a record-amplification cap. Its bias can be plugged into Nakada et al.'s bias term.
2. **The sampler.** Field-wise twisted SMC on an autoregressive row writer, with a partial-row twist learned from masked rows: FUDGE adapted to table fields.
3. **Sequence-level contrast guidance** between minority- and majority-conditioned rows, shown to be a special case of the same tilt.
4. **The theory.** It separates "a perfect generator only equals a tuned threshold" (Prop. 1) from "a KL-budgeted tilt has a first-order finite-sample gain" (Prop. 2). The evaluation is designed around it: `gain_vs_threshold` is the headline metric.

## What reviewers will say is already known, and our answer

| objection | answer |
|---|---|
| "Classifier-feedback tilting already exists." Askari-Hemmat et al., TMLR 2024, 2310.00158, write the same tilt for **diffusion on images** (loss, entropy or hardness guidance). | Cite it as the closest prior work. Ours differs in data type (tables), generator (fine-tuned LLM, discrete fields), sampler (SIR / twisted SMC, not gradients), the explicit KL budget, the validity floor, and privacy. |
| "Boundary emphasis is old": Borderline-SMOTE, ADASYN, GAMO (ICCV 2019), GAAL, LiBaGS (2605.11231) | Borderline-SMOTE and SVM-SMOTE are baselines in `thesis_main`. A LiBaGS-style hard selection is covered by `tilt_aleatoric` with a large budget. Our utility is *derived* (Prop. 2), not heuristic. |
| "exp(λr) is the KL-regularised optimum; everyone knows" (RLHF, TERM ICLR 2021, KL-DRO) | Yes, Lemma 1 is cited as KNOWN. The contribution is the choice of \(u\) (Prop. 2) and the proof that it cannot help at population level (Prop. 1). |
| "A perfect generator ≡ reweighting" (Nakada et al. 2406.03628) | Agreed; that is Prop. 1's premise. TILT is *deliberate* bias. |
| "Twisted SMC and FUDGE are off the shelf" | They are, for text. We found no use on tabular rows. The contribution is the application and its checks. |
| "Tuning the threshold matches oversampling" (Elor & Averbuch-Elor 2201.08528) | That is our anchor baseline. We report `gain_vs_threshold` for every method. |
| **"Uncertain rows labelled minority are label noise"** (the CLLM argument), the most likely attack | The default utility is **variance** (fisher / epistemic), not ambiguity. A non-linear **validity floor** rejects rows less minority-like than 95% of real minority rows. `tilt_aleatoric` is a named ablation predicted to hurt (H4). The `validity` experiment switches the floor off to measure its effect. |

## Closest works

Abbreviations: T = theory, Tilt = utility-tilted target, SMC = sequential or twisted SMC, LLM = LLM tabular generation, Imb = imbalance focus.

| work | what it does | differs from TILT by | T | Tilt | SMC | LLM | Imb |
|---|---|---|---|---|---|---|---|
| Askari-Hemmat et al., TMLR 2024 (2310.00158) | Diffusion guidance by classifier loss / entropy / hardness | images; gradient guidance; no KL budget, SMC or privacy | partial | **yes** | – | – | yes |
| Nakada et al. (2406.03628, v3 2026) | Synthetic oversampling theory; θ_bal; transformer generator | no utility signal; bias is only harmful | yes | – | – | yes | yes |
| ImbLLM (2510.09783) | Conditioned LLM oversampling | no classifier feedback | – | – | – | yes | yes |
| LiBaGS (2605.11231) | Heuristic selection by boundary / uncertainty / density | not derived; not LLM | – | heuristic | – | – | ? |
| LITO (ICLR 2024) | LLM imputes "punctured" majority rows under the minority label | implicit borderline rows | – | implicit | – | yes | yes |
| EPIC (NeurIPS 2024, 2404.12404) | Balanced in-context prompting | no feedback | – | – | – | yes | yes |
| CLLM (ICML 2024, 2312.12112) | Curation by learning dynamics | *removes* ambiguous rows (the opposite sign) | – | inverse | – | yes | low-data |
| ReTabSyn (2603.10823) | DPO on a GReaT generator | rewards p(y\|x) fidelity; training-time | – | – | – | yes | eval |
| TabGRAA (2604.18966) | Reward-guided post-training (discriminator, DCR) | fidelity and privacy, not utility | – | RL | – | yes | – |
| P-TA (2406.11391) | PPO with a GAN discriminator reward | realism reward | – | – | – | yes | – |
| GAMO (ICCV 2019) | Adversarial convex generator | not LLM; no KL control | – | adversarial | – | – | yes |
| UGO (KBS 2026); LEO-CVAE (2509.25334) | Uncertainty-driven seeding / weighting | no LLM, no tilt objective | – | partial | – | – | yes |
| Overlap class (AAAI 2025, 2412.15657) | Third class for the overlap region | relabelling, not guidance | – | – | – | – | yes |
| Twisted SMC (Zhao et al., ICML 2024, 2404.17546) | SMC for LMs, learned twists | text only | yes | general | **yes** | – | – |
| FUDGE; CFG-LM (2306.17806); Controlled decoding (2310.17022); SMC steering (2306.03081, 2504.13139) | Decoding-time control with partial-sequence discriminators | never applied to table rows | – | – | yes | – | – |
| TERM (ICLR 2021, 2007.01162); MMEL (ICLR 2021) | Exponential tilting of losses / reweighting | reweights fixed samples; does not sample | yes | weights | – | – | – |

## Must cite

**LLM tabular generation:**
- GReaT (2210.06280)
- Pred-LLM (2410.21717)
- ImbLLM
- Nakada et al.
- LITO, EPIC, CLLM
- ReTabSyn, TabGRAA, P-TA

**Classifier feedback and boundary emphasis:**
- Askari-Hemmat et al.
- LiBaGS
- GAMO
- SMOTE, Borderline-SMOTE, ADASYN
- SOS (KDD 2022, 2206.08555)

**Other generators and imbalance fixes:**
- TabDDPM (2209.15421)
- TabEBM (2409.16118)
- the overlap class (AAAI 2025)

**Guided decoding and SMC:**
- FUDGE
- twisted SMC
- CFG-LM
- Controlled Decoding

**Theory and tilting:**
- TERM
- Elkan 2001
- Koyejo et al. 2014
- Menon et al. 2021
- Koh & Liang 2017

**Thresholds and privacy:**
- Elor & Averbuch-Elor
- Carlini et al. 2022
- Stadler et al. 2022
- Yao et al. 2025 (DCR delusion)

## Public code usable as baselines

| baseline | how to get it |
|---|---|
| imbalanced-learn (SMOTE family) | `pip` |
| CTGAN / TVAE | SDV |
| TabDDPM / ARF | synthcity (separate env) |
| GReaT | `pip install be-great` |
| CLLM | seedatnabeel/CLLM; this repo also has a CLLM-style curator |
| EPIC | seharanul17/synthetic-tabular-LLM; can run on a local Qwen-Instruct |
| TabEBM | `pip install tabebm` |
| SOS | JayoungKim408/SOS |
| Feedback-guided synthesis | facebookresearch/Feedback-guided-Data-Synthesis |
| TabDiff | MinkaiXu/TabDiff |
| TabSyn | amazon-science/tabsyn |

No public code was found for ImbLLM (palt3 re-implements it), LITO, LiBaGS, or UGO.
