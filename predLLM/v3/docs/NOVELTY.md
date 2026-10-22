# v3: Negative Alignment & Outlier-Bounded LLM Tabular Generation

v3 extends Pred-LLM (v1) and schema-constrained decoding (v2) to address the central failure modes of imbalanced and rare-event tabular generation:
1. **Majority leakage / mode collapse**: Standard LLM generators drift towards the dominant majority distribution even when conditioned on rare labels.
2. **Support unboundedness & tail neglect**: Standard decoders either hallucinate values into non-existent feature space or fail to cover the rare/anomalous tail manifold where minority points live.

Each v3 component can be toggled independently as an ablation (`python -m predllm_v3 ablation`).

---

## N1. Negative Majority-Conditioned Unlikelihood Fine-Tuning (Objective)

**Problem.** In imbalanced tables (e.g. fraud, medical disease, equipment failure), the majority class comprises 90–99.9% of the data. When fine-tuning an autoregressive LLM, token statistics are overwhelmingly shaped by majority patterns. Merely conditioning on `"target is 1"` often fails because common majority combinations have higher autoregressive probability mass.

**v3 Solution.** We divide the training set $D$ into target minority rows $D_{\text{pos}}$ and majority contrast rows $D_{\text{neg}}$.
The training objective optimizes:

$$L_{v3}(\theta) = L_{\text{pos}}(D_{\text{pos}}) + \beta_{\text{neg}} L_{\text{unlikelihood}}(D_{\text{neg}})$$

where:
* **Positive Loss ($L_{\text{pos}}$)**: Standard target-weighted value-only cross entropy:
  $$L_{\text{pos}} = - \frac{\sum_{i \in \text{pos}} \sum_t w_t \log p_\theta(c_t \mid c_{<t})}{\sum_{i \in \text{pos}} \sum_t w_t}$$
* **Negative Unlikelihood Loss ($L_{\text{unlikelihood}}$)**: Explicitly penalizes generating majority tokens:
  $$L_{\text{unlikelihood}} = - \frac{\sum_{j \in \text{neg}} \sum_t w_t \log (1 - p_\theta(c_t \mid c_{<t}) + \epsilon)}{\sum_{j \in \text{neg}} \sum_t w_t}$$

As $p_\theta(c_t \mid c_{<t}) \to 0$ on majority samples, $-\log(1) = 0$ (no penalty). As the model assigns probability to majority configurations, $-\log(1 - p) \to \infty$, forcefully repelling the generator away from majority density.

Ablation: `v3_no_negative` (`finetune.negative_weight=0.0`).

---

## N2. Outlier-Aware Support Bounding (Decoding & Support Envelopes)

**Problem.** Standard numerical generation clips to the global training bounds $[\min(x_j), \max(x_j)]$. In imbalanced or anomalous tables, the rare/minority events are localized in specific tail regions (the "outlier envelope" of the majority distribution). Generating across the entire global range produces points with no relevance to the decision boundary.

**v3 Solution.** For each numeric feature $X_j$, v3 computes the **outlier-aware generation envelope** using configurable statistical boundaries:
* **IQR Envelope (`mode='iqr'`)**: $[Q_1 - k \cdot \text{IQR}, Q_3 + k \cdot \text{IQR}]$ identifying the boundary thresholds.
* **Minority Support Envelope (`mode='minority'`)**: Strict empirical bounds $[\min(x_{j, \text{minority}}), \max(x_{j, \text{minority}})]$.
* **Extreme Quantile Envelope (`mode='quantile'`)**: $[q_{0.01}, q_{0.99}]$.

At decoding time, the finite-state machine (FSM) masks digits and clamps candidate completions specifically to the active envelope `sch.active_range(col)`. Synthetic values are guaranteed 100% valid and concentrated in the target rare-event manifold.

Ablation: `v3_no_outlier_range` (`range_mode=empirical`).

---

## N3. Anomaly-Gated Multi-Stage Curation (Post-Processing)

**Problem.** Rejection sampling on distance alone (DCR) only checks if a row is an exact duplicate. It does not check if a generated row has collapsed back into the dense majority cluster.

**v3 Solution.** A 3-stage post-hoc curation pipeline:
1. **Privacy Gate (DCR)**: Drops near-copies closer than $\tau = \text{Quantile}_\alpha(\text{DCR})$ to real records.
2. **Anomaly Filter**: An `IsolationForest` scores candidate synthetic rows. Rows with anomaly scores falling into the dense majority inlier cluster are pruned, ensuring only points that respect the outlier profile are retained.
3. **Discriminator SIR**: Sampling-importance-resampling via cross-fitted XGBoost density-ratio correction.

Ablation: `v3_no_curation` (`curation.enabled=false`).

---

## Summary of Novelties

| Component | Pred-LLM (v1) | v2 | v3 | Module |
|---|---|---|---|---|
| **Majority Signal** | Ignored (trained jointly) | Positive loss on all tokens | **Negative unlikelihood penalty ($\beta_{\text{neg}}$)** | `finetune.py` |
| **Numerical Range** | Paper MinMax $[0, 1]$ | Global empirical $[\min, \max]$ | **Outlier-aware support envelope (IQR/Quantile)** | `outlier.py`, `codec.py` |
| **Decoding Constraint** | Free text + parse rejection | Global FSM | **Outlier-bounded FSM + CFG negative guidance** | `decoding.py`, `sampler.py` |
| **Curation** | None | DCR gate + SIR | **DCR gate + Anomaly filter + SIR** | `curation.py` |
