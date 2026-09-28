"""
predllm_v1: faithful replication of Pred-LLM (Nguyen et al., arXiv:2410.21717).

methods    Table III presets (pred_llm, great, taptap, ablation rows)
serialize  textual encoding "X is v", permute_x / permute_xy, parsing       (Sec. III-B1a,b)
finetune   D'_real = permuted U original sentences (Alg. 1 l.2-12)            (Sec. III-B1c)
sampler    feature- / target-conditional free-text sampling  [LLM]          (Sec. III-B2)
querier    label querying via LLM prompt / XGBoost / none   [LLM]          (Sec. III-B3)
generator  wires the three phases + model cache (Alg. 1)
cli        run / benchmark / ablation / train / generate / infer / evaluate / attention / smoke
"""

from typing import Any, Optional, List, Dict, Union
