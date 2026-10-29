"""THE MODEL SWITCH -- every backbone this repo may fine-tune, and nothing else.

A backbone is listed only if it satisfies all of:
  1. decoder-only causal LM with a Hugging Face AutoModelForCausalLM class
     (the method needs left-to-right "prompt -> complete the row" sampling)
  2. a BASE (pre-trained, not chat-tuned) checkpoint: fine-tuning a chat
     template onto "col is value" rows fights the instruction tuning
  3. fine-tunable with full / LoRA / QLoRA on one GPU from 6 GB to 80 GB
     (the mode is chosen per GPU by core/llm/hardware.py, not by you)
  4. no trust_remote_code

Anything else is rejected with the reason (see INCOMPATIBLE).

`n_layer` / `hidden` feed the activation-memory estimate; `min_transformers`
is checked at load time so a too-old library fails with a clear message.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    params_b: float
    family: str
    n_layer: int
    hidden: int
    gated: bool = False
    min_transformers: str = "4.40"
    notes: str = ""


MODELS: dict[str, ModelSpec] = {m.key: m for m in [
    # offline test backbone: random init, byte-level tokenizer, built locally
    ModelSpec("tiny-random", "local:tiny-random", 0.001, "tiny", 4, 128,
              notes="Offline smoke tests only. Never report numbers from it."),
    # GPT-2 family -- the paper's backbone is distilgpt2
    ModelSpec("distilgpt2", "distilbert/distilgpt2", 0.082, "gpt2", 6, 768,
              notes="THE PAPER'S BACKBONE. Use for every v1 replication number."),
    ModelSpec("gpt2", "openai-community/gpt2", 0.124, "gpt2", 12, 768),
    ModelSpec("gpt2-medium", "openai-community/gpt2-medium", 0.355, "gpt2", 24, 1024),
    ModelSpec("gpt2-large", "openai-community/gpt2-large", 0.774, "gpt2", 36, 1280),
    # SmolLM2 -- small, ungated
    ModelSpec("smollm2-135m", "HuggingFaceTB/SmolLM2-135M", 0.135, "llama", 30, 576),
    ModelSpec("smollm2-360m", "HuggingFaceTB/SmolLM2-360M", 0.36, "llama", 32, 960),
    ModelSpec("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B", 1.7, "llama", 24, 2048),
    # Qwen2.5 base -- ungated size ladder
    ModelSpec("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B", 0.49, "qwen2", 24, 896),
    ModelSpec("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B", 1.54, "qwen2", 28, 1536),
    ModelSpec("qwen2.5-3b", "Qwen/Qwen2.5-3B", 3.09, "qwen2", 36, 2048),
    ModelSpec("qwen2.5-7b", "Qwen/Qwen2.5-7B", 7.6, "qwen2", 28, 3584),
    # Qwen3 base -- ungated, needs transformers >= 4.51
    ModelSpec("qwen3-0.6b", "Qwen/Qwen3-0.6B-Base", 0.6, "qwen3", 28, 1024,
              min_transformers="4.51"),
    ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B-Base", 1.7, "qwen3", 28, 2048,
              min_transformers="4.51"),
    ModelSpec("qwen3-4b", "Qwen/Qwen3-4B-Base", 4.0, "qwen3", 36, 2560,
              min_transformers="4.51"),
    ModelSpec("qwen3-8b", "Qwen/Qwen3-8B-Base", 8.2, "qwen3", 36, 4096,
              min_transformers="4.51"),
    # Llama 3.x base -- GATED: accept the licence on the model page + HF_TOKEN
    ModelSpec("llama3.2-1b", "meta-llama/Llama-3.2-1B", 1.24, "llama", 16, 2048,
              gated=True, min_transformers="4.45"),
    ModelSpec("llama3.2-3b", "meta-llama/Llama-3.2-3B", 3.21, "llama", 28, 3072,
              gated=True, min_transformers="4.45"),
    ModelSpec("llama3.1-8b", "meta-llama/Llama-3.1-8B", 8.03, "llama", 32, 4096,
              gated=True, min_transformers="4.43"),
    # Mistral 7B base
    ModelSpec("mistral-7b", "mistralai/Mistral-7B-v0.1", 7.24, "mistral", 32, 4096),
]}

DEFAULT = "distilgpt2"

INCOMPATIBLE: dict[str, str] = {
    "bert": "encoder-only: cannot autoregressively sample a row",
    "roberta": "encoder-only: cannot autoregressively sample a row",
    "deberta": "encoder-only: cannot autoregressively sample a row",
    "t5": "encoder-decoder: the pipeline is prompt->continuation, not seq2seq",
    "bart": "encoder-decoder: the pipeline is prompt->continuation, not seq2seq",
    "gpt-4": "API-only: cannot be fine-tuned locally",
    "gpt-4o": "API-only: cannot be fine-tuned locally",
    "claude": "API-only: cannot be fine-tuned locally",
    "gemini": "API-only: cannot be fine-tuned locally",
    "instruct": "chat/instruct checkpoint: use the base model of the same size",
    "-chat": "chat/instruct checkpoint: use the base model of the same size",
    "openelm": "needs trust_remote_code and a separate tokenizer; excluded",
}


def get(key: str) -> ModelSpec:
    k = key.lower().strip()
    if k in MODELS:
        return MODELS[k]
    for bad, why in INCOMPATIBLE.items():
        if bad in k:
            raise ValueError(f"{key!r} is not usable here: {why}.\n"
                             f"Compatible: {', '.join(MODELS)}")
    raise ValueError(f"Unknown model {key!r}. Compatible: {', '.join(MODELS)}")


def check_access(spec: ModelSpec) -> None:
    if spec.gated and not (os.environ.get("HF_TOKEN")
                           or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        raise PermissionError(
            f"{spec.key} ({spec.hf_id}) is gated: accept the licence on its Hub page, "
            f"then `export HF_TOKEN=hf_...` (or `huggingface-cli login`).")
    if spec.family == "tiny":
        return
    try:
        import transformers
        from packaging.version import Version
        if Version(transformers.__version__) < Version(spec.min_transformers):
            raise RuntimeError(f"{spec.key} needs transformers>={spec.min_transformers}, "
                               f"found {transformers.__version__}")
    except ImportError:
        raise RuntimeError('LLM extra missing: pip install -e ".[llm]"') from None
