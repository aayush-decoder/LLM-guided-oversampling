"""
LLM switch: resolves `--llm` (alias / registry key / raw HF id) into an LLMSpec.
"""

from typing import Any, Optional, List, Dict, Union

class LLMSpec:
    @property
    def tag(self):
        """Filesystem-safe name used in run directories."""
        pass

def _registry():
    pass

def list_llms():
    pass

def resolve_llm(name, mode):
    """`name` may be an alias ('1.5b'), a registry key ('qwen2.5-1.5b') or any HF causal-LM id.
Unregistered HF ids are allowed but unverified (defaults to LoRA)."""
    pass
