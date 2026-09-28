"""
Module: common.tabgen.llm
Responsibility: LLM model registry, loader (full/LoRA/QLoRA), and training loop abstractions.
"""
from typing import Any

class LLMSpec:
    """Specification for an LLM backbone."""
    pass

def list_llms(*args, **kwargs):
    """List all registered LLM configurations."""
    pass

def resolve_llm(*args, **kwargs):
    """Resolve an LLM alias or name to an LLMSpec."""
    pass

__all__ = ["LLMSpec", "list_llms", "resolve_llm"]
