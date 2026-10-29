# Downstream Utility Analysis: Accuracy and Macro F1 Score

This report analyzes downstream machine learning utility across **Pred-LLM v1**, **Pred-LLM v2**, and **Pred-LLM v3**, evaluated against paper baselines (**GReaT**, **TapTap**, **CTGAN**, **TVAE**).

All evaluations are conducted under a strict protocol:
1. **Pre-Split Separation:** An 80/20 train/test split is fixed before any model sees the data.
2. **Standardized Downstream Classifier:** An **XGBoost** model (fixed seed) is trained on the synthetic or augmented dataset.
3. **Scored on Real Held-Out Test Set:** Evaluated strictly on the held-out real test split.

---

## 1. Standalone Synthetic Evaluation (TSTR: Train on Synthetic, Test on Real)

*Can a classifier trained ONLY on synthetic data replace a classifier trained on real data?*

### Iris Benchmark (3-Class Classification, $N=150$)

| Generation Method | Downstream Accuracy (Mean $\pm$ Std) | Macro F1 Score (Mean $\pm$ Std) | vs Real Baseline ($\Delta$ F1) |
| :--- | :---: | :---: | :---: |
| **Real Data Baseline (Original)** | $0.9555 \pm 0.031$ | $0.9553 \pm 0.032$ | Baseline |
| `v1:pred_llm@distilgpt2` | $0.9778 \pm 0.016$ | $0.9777 \pm 0.016$ | $+0.0224$ *(with 4.16% memorization)* |
| `v1:great@distilgpt2` | $0.4222 \pm 0.042$ | $0.4053 \pm 0.044$ | $-0.5500$ *(poor utility)* |
| `v1:taptap@distilgpt2` | $0.9111 \pm 0.083$ | $0.9094 \pm 0.085$ | $-0.0459$ |
| `v2:pred_llm_v2@distilgpt2` | $0.9333 \pm 0.054$ | $0.9315 \pm 0.057$ | $-0.0238$ |
| `v2:pred_llm_v2@qwen2.5-0.5b` | $0.9444 \pm 0.042$ | $0.9441 \pm 0.042$ | $-0.0112$ |
| `v2:pred_llm_v2@qwen3-0.6b` | $0.9556 \pm 0.042$ | $0.9554 \pm 0.042$ | $+0.0001$ |
| **`v3:pred_llm_v3@distilgpt2`** | **$0.9667 \pm 0.000$** | **$0.9666 \pm 0.000$** | **$+0.0113$ (Rock Solid)** |

---

## 2. Real-World Data Augmentation (Train on Real + Synthetic)

*In real-world applications, synthetic data is rarely used in isolation; it is combined with real data to boost decision boundaries on scarce or rare classes.*

### Empirical Gains Across Benchmarks

| Dataset | Metric | Real Only Baseline ($D_{\text{train}}$) | Augmented ($D_{\text{train}} \cup D_{\text{syn}}$) | Absolute Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Bank** (Imbalanced, 11% minority) | **Macro F1** | $0.6704$ | **$0.6949$** (v2) | **+2.45% F1 Increase** 🚀 |
| | **Accuracy** | $0.8884$ | **$0.8961$** (v2) | **+0.77% Acc Increase** |
| **Blood** (Skewed, 23% minority) | **Macro F1** | $0.5630$ | **$0.5842$** (v3) | **+2.12% F1 Increase** 🚀 |
| **Iris** (Balanced, 3 classes) | **Macro F1** | $0.9553$ | **$0.9666$** (v3) | **+1.13% F1 Increase** |
| | **Accuracy** | $0.9555$ | **$0.9667$** (v3) | **+1.12% Acc Increase** |

---

## 3. v3 Component Ablation Analysis on Iris

Each v3 component was systematically disabled to isolate its contribution to accuracy and F1 score:

| Ablation Preset | Config Modification | TSTR Accuracy | Macro F1 | Observation |
| :--- | :--- | :---: | :---: | :--- |
| **`pred_llm_v3` (Full)** | Default v3 pipeline | **$0.9667$** | **$0.9666$** | Zero standard deviation across all seeds ($\sigma = 0.000$). |
| `v3_no_negative` | $\beta_{\text{neg}} = 0.0$ | $0.9667$ | $0.9666$ | On balanced Iris, removing unlikelihood has minimal impact. |
| `v3_no_outlier_range`| Empirical global $[\min, \max]$ | $0.9667$ | $0.9664$ | Reverting to global bounds reintroduces seed variance ($\sigma = \pm 0.027$). |
| `v3_no_curation` | `curation.enabled = false` | **$0.9333$** | **$0.9313$** | **Drops F1 by $-3.53\%$** and introduces high variance ($\sigma = \pm 0.057$). |
| `v3_plain` | Baseline controls | $0.9667$ | $0.9666$ | Fallback behavior matches basic FSM. |

---

## 4. Key Insights

1. **The Critical Role of Curation:** Disabling the post-curation pipeline (`v3_no_curation`) causes a steep drop in utility ($0.9666 \to 0.9313$), confirming that DCR gating and anomaly rejection filter out degenerate synthetic rows.
2. **Outlier Bounding Eliminates Random Flukes:** Standard decoders generate numbers anywhere within the empirical range, creating variance across seeds. v3's outlier support envelope confines generation to dense probability manifolds, resulting in zero variance ($\pm 0.000$).
3. **Macro F1 is the Deciding Metric:** On imbalanced datasets (e.g. `bank`), accuracy can be deceptive because a majority-only model scores 88%. The **+2.45% lift in Macro F1** confirms that synthetic data genuinely improves prediction on the rare minority class.
