# Loads (tokenizer, model) for an LLMSpec in full / LoRA / QLoRA mode, and saves/restores fine-tuned
# weights. This is the ONLY place a pretrained LLM is instantiated.
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils import get_logger, load_json, save_json
from .registry import LLMSpec

log = get_logger("tabgen.llm")


def _dtype(device: str, mode: str):
    if device.startswith("cuda") and mode in ("lora", "qlora"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32     # full fine-tuning keeps fp32 master weights (autocast handles speed)


def load_tokenizer(hf_id: str):
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.eos_token_id is None:
        raise ValueError(f"{hf_id} has no EOS token; not supported")
    return tok


def vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def load_llm(spec: LLMSpec, device: str = "cuda", trainable: bool = True, attn: str | None = None,
             gradient_checkpointing: bool | None = None):
    """Returns (tokenizer, model). With trainable=True, LoRA/QLoRA adapters are attached."""
    if device.startswith("cuda") and spec.min_vram_gb and vram_gb() < spec.min_vram_gb:
        log.warning(f"{spec.key} suggests >= {spec.min_vram_gb} GB VRAM, GPU has {vram_gb():.1f} GB")
    tok = load_tokenizer(spec.hf_id)
    kwargs = {"torch_dtype": _dtype(device, spec.mode)}
    if attn:
        kwargs["attn_implementation"] = attn
    if spec.mode == "qlora":
        if not device.startswith("cuda"):
            raise RuntimeError("QLoRA needs a CUDA GPU (bitsandbytes). Use --mode lora on CPU.")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=kwargs["torch_dtype"])
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
    if spec.mode != "qlora":
        model.to(device)
    model.config.pad_token_id = tok.pad_token_id
    if trainable and spec.mode in ("lora", "qlora"):
        model = _attach_lora(model, spec, gradient_checkpointing)
    return tok, model


def _attach_lora(model, spec: LLMSpec, gradient_checkpointing: bool | None):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    gc = spec.mode == "qlora" if gradient_checkpointing is None else gradient_checkpointing
    if spec.mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gc)
    elif gc:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    cfg = LoraConfig(r=spec.lora["r"], lora_alpha=spec.lora["alpha"], lora_dropout=spec.lora["dropout"],
                     target_modules=spec.lora["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log.info(f"LoRA attached: {n_train/1e6:.2f}M / {n_all/1e6:.1f}M trainable params")
    return model


def save_finetuned(model, tok, spec: LLMSpec, out_dir: str | Path, extra: dict | None = None) -> None:
    """Full mode saves the whole model; LoRA/QLoRA modes save only the adapter (a few MB)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    save_json({"llm_key": spec.key, "hf_id": spec.hf_id, "mode": spec.mode, "lora": spec.lora,
               **(extra or {})}, out_dir / "llm_meta.json")


def load_finetuned(model_dir: str | Path, device: str = "cuda", attn: str | None = None):
    """Restores (tokenizer, model, meta) saved by save_finetuned, ready for inference."""
    model_dir = Path(model_dir)
    meta = load_json(model_dir / "llm_meta.json")
    spec = LLMSpec(key=meta["llm_key"], hf_id=meta["hf_id"], mode=meta["mode"], lora=meta["lora"])
    if spec.mode == "full":
        tok = load_tokenizer(str(model_dir))
        kw = {"attn_implementation": attn} if attn else {}
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, **kw).to(device)
    else:
        from peft import PeftModel
        tok, base = load_llm(spec, device=device, trainable=False, attn=attn)
        model = PeftModel.from_pretrained(base, model_dir)
    model.eval()
    return tok, model, meta
