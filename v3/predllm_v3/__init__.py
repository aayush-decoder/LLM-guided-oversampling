"""predllm_v3: Pred-LLM with majority negative unlikelihood fine-tuning, outlier-aware
support range bounding, and anomaly score curation. See v3/docs/NOVELTY.md.

outlier    outlier/boundary range computation (IQR/quantiles) + IsolationForest anomaly scoring
codec      schema with outlier envelopes, canonical piece-wise tokenization
finetune   V3RowDataset with positive (minority) and negative (majority) unlikelihood training
decoding   exact candidate scoring + numeric FSM decoding + CFG negative logit guidance
sampler    field-wise feature sampling constrained to the active outlier support range
posterior  permutation-marginalised p(y|x) exact scoring with optional minority prior tilt
curation   DCR privacy gate + IsolationForest anomaly filter + discriminator SIR
methods    presets / ablations (pred_llm_v3, v3_no_negative, v3_no_outlier_range, ...)
generator  PredLLMV3 wires everything with model caching
cli        run / train / generate / infer / evaluate / smoke / methods / ablation
"""
