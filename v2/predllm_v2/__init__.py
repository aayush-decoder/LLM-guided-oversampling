"""
predllm_v2: Pred-LLM with schema-constrained decoding, exact permutation-marginalised label
posteriors, weighted value-only training and discriminator-guided curation. See v2/docs/NOVELTY.md.

codec      schema (raw categories/units), canonical piece-wise tokenization, loss weights   (N1, N3)
finetune   D'_real with per-token weights
decoding   exact candidate scoring (KV cache) + numeric FSM decoding          [LLM]   (N2)
sampler    field-wise feature-conditional sampling of x                     [LLM]   (N2)
posterior  permutation-marginalised p(y|x), regression by constrained draws [LLM]   (N4)
curation   DCR privacy gate + discriminator SIR                                     (N5)
methods    presets / ablations (config overrides)
generator  wires everything + model cache
cli        run / benchmark / ablation / train / generate / infer / evaluate / smoke
"""

from typing import Any, Optional, List, Dict, Union
