"""
Loads (tokenizer, model) for an LLMSpec in full / LoRA / QLoRA mode, and saves/restores fine-tuned
weights. This is the ONLY place a pretrained LLM is instantiated.
"""

from typing import Any, Optional, List, Dict, Union

def _dtype(device, mode):
    pass

def load_tokenizer(hf_id):
    pass

def vram_gb():
    pass

def load_llm(spec, device, trainable, attn, gradient_checkpointing):
    """Returns (tokenizer, model). With trainable=True, LoRA/QLoRA adapters are attached."""
    pass

def _attach_lora(model, spec, gradient_checkpointing):
    pass

def save_finetuned(model, tok, spec, out_dir, extra):
    """Full mode saves the whole model; LoRA/QLoRA modes save only the adapter (a few MB)."""
    pass

def load_finetuned(model_dir, device, attn):
    """Restores (tokenizer, model, meta) saved by save_finetuned, ready for inference."""
    pass
