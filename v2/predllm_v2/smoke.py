"""
v2 smoke test: iris, 1 seed, short fine-tune; pred_llm_v2 + two ablations + Original.
Works even with the random tiny-gpt2, because v2's decoder cannot emit invalid rows.
"""

from typing import Any, Optional, List, Dict, Union

def run_smoke(llm, device, max_steps):
    pass
