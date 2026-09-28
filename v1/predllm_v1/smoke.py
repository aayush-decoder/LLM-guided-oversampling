"""
v1 smoke test: iris, 1 seed, short fine-tune; Pred-LLM + GReaT + TapTap + Original (+ CTGAN if SDV).

v1 decodes free text, so the LLM must learn the "X is v" format before it can emit valid rows.
The default is therefore distilgpt2 with ~150 steps (seconds on GPU, a few minutes on CPU);
tiny-gpt2 is too small to learn the format (v2 has no such limit, since its decoder is schema-constrained).
"""

from typing import Any, Optional, List, Dict, Union

def run_smoke(llm, device, max_steps):
    pass
