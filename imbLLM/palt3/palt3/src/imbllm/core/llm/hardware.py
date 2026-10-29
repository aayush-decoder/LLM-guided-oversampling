"""Hardware profiles and the training PLAN: how a given backbone runs on a given GPU.

The user picks a model; this module picks everything that depends on the
device, so the same command works on a 6 GB laptop GPU and on an 80 GB H100:

    mode          full | lora | qlora     (largest-capacity mode that fits)
    precision     bf16 | fp16 | fp32      (bf16 on Ampere+; fp16 + loss scaling
                                           on Turing/Pascal; fp32 on CPU)
    micro_batch   largest power of two that fits;  grad_accum keeps the
                  EFFECTIVE batch at the paper's 32 on every device
    grad_ckpt     on when activations would not fit otherwise
    attn          sdpa everywhere; flash_attention_2 on bf16 GPUs if installed

Precision rule (the palt2 crash, fixed): trainable parameters are ALWAYS fp32.
Full fine-tuning keeps fp32 master weights and autocasts the forward pass;
LoRA/QLoRA freeze a bf16/fp16/4-bit base and train fp32 adapters. fp16 uses a
GradScaler; bf16 does not need one.

Memory model (GB, deliberately conservative):
    full   16 * P            fp32 weights + grads + Adam m, v
    lora    2 * P + 0.5      frozen half-precision base + adapters/optimizer
    qlora   0.6 * P + 0.5    4-bit NF4 base
    + activations  micro_batch * seq * n_layer * hidden * k bytes
                   (k = 34 half precision, ~ 6 with gradient checkpointing)
    + 1.0 CUDA context / fragmentation
Profiles list usable VRAM (a bit under the card's nominal size).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ... import paths
from ...config import load_yaml
from .zoo import ModelSpec


@dataclass
class Hardware:
    name: str
    device: str                  # cuda | mps | cpu
    vram_gb: float               # usable memory for this process
    bf16: bool
    qlora: bool                  # bitsandbytes 4-bit usable (CUDA cc >= 7.0)
    full_ft_max_b: float = 99.0  # never full-fine-tune above this many B params
    flash_attn: bool = False
    gen_batch: int = 64

    def as_dict(self) -> dict:
        return asdict(self)


def load_profile(name: str) -> Hardware:
    d = load_yaml(name, paths.HARDWARE_CONFIGS)
    return Hardware(**d)


def detect() -> Hardware:
    """Best-effort description of THIS machine."""
    try:
        import torch
    except ImportError:
        return Hardware("cpu (no torch)", "cpu", 0.0, False, False, full_ft_max_b=0.4,
                        gen_batch=16)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        cc = p.major + p.minor / 10
        vram = p.total_memory / 1024 ** 3 * 0.92
        try:
            import flash_attn  # noqa: F401
            fa = cc >= 8.0
        except ImportError:
            fa = False
        return Hardware(f"{p.name} (detected)", "cuda", round(vram, 1),
                        bf16=bool(torch.cuda.is_bf16_supported()) and cc >= 8.0,
                        qlora=cc >= 7.0, flash_attn=fa,
                        gen_batch=16 if vram < 8 else 64 if vram < 30 else 256)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return Hardware("apple mps (detected)", "mps", 8.0, False, False,
                        full_ft_max_b=0.4, gen_batch=16)
    return Hardware("cpu (detected)", "cpu", 0.0, False, False, full_ft_max_b=0.4,
                    gen_batch=16)


def get_hardware(name: str | None) -> Hardware:
    return detect() if name in (None, "", "auto") else load_profile(name)


@dataclass
class TrainPlan:
    model: str
    hardware: str
    device: str
    mode: str                    # full | lora | qlora
    precision: str               # bf16 | fp16 | fp32
    micro_batch: int
    grad_accum: int
    effective_batch: int
    grad_ckpt: bool
    attn: str
    est_vram_gb: float
    fits: bool
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def _static_gb(spec: ModelSpec, mode: str) -> float:
    p = spec.params_b
    return {"full": 16 * p, "lora": 2 * p + 0.5, "qlora": 0.6 * p + 0.5}[mode]


def _act_gb(spec: ModelSpec, mb: int, seq: int, ckpt: bool) -> float:
    k = 6 if ckpt else 34
    return mb * seq * spec.n_layer * spec.hidden * k / 1024 ** 3


def plan(spec: ModelSpec, hw: Hardware, *, mode: str = "auto", effective_batch: int = 32,
         seq_len: int = 256) -> TrainPlan:
    """Choose mode/precision/micro-batch for `spec` on `hw`. Never silently OOMs:
    `fits=False` comes with a reason, and the engine refuses to start."""
    precision = "bf16" if hw.bf16 else ("fp16" if hw.device == "cuda" else "fp32")
    attn = "flash_attention_2" if (hw.flash_attn and precision == "bf16") else "sdpa"

    if hw.device != "cuda":
        # CPU / MPS: full fine-tuning of small models only, fp32.
        ok = spec.params_b <= hw.full_ft_max_b or spec.family == "tiny"
        mb = min(effective_batch, 8)
        return TrainPlan(spec.key, hw.name, hw.device, "full", "fp32", mb,
                         max(1, effective_batch // mb), mb * max(1, effective_batch // mb),
                         False, "sdpa", 16 * spec.params_b, ok,
                         "cpu/mps: full fine-tune in fp32" if ok else
                         f"{spec.key} is too large for {hw.device}; use <= "
                         f"{hw.full_ft_max_b}B or a GPU")

    modes = ["full", "lora", "qlora"] if mode == "auto" else [mode]
    tried = []
    for m in modes:
        if m == "full" and spec.params_b > hw.full_ft_max_b:
            tried.append(f"full: > {hw.full_ft_max_b}B cap for this profile")
            continue
        if m == "qlora" and not hw.qlora:
            tried.append("qlora: bitsandbytes 4-bit needs CUDA compute capability >= 7.0")
            continue
        static = _static_gb(spec, m) + 1.0
        for ckpt in (False, True):
            mb = effective_batch
            while mb >= 1:
                total = static + _act_gb(spec, mb, seq_len, ckpt)
                if total <= hw.vram_gb:
                    accum = max(1, effective_batch // mb)
                    return TrainPlan(spec.key, hw.name, hw.device, m, precision, mb, accum,
                                     mb * accum, ckpt, attn, round(total, 2), True,
                                     f"{m} fits: ~{total:.1f} of {hw.vram_gb} GB")
                mb //= 2
        tried.append(f"{m}: needs >= {static + _act_gb(spec, 1, seq_len, True):.1f} GB")
    return TrainPlan(spec.key, hw.name, hw.device, modes[-1], precision, 1, effective_batch,
                     effective_batch, True, attn, float("nan"), False,
                     f"{spec.key} does not fit {hw.name} ({hw.vram_gb} GB): "
                     + "; ".join(tried))
