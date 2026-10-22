# LLM switch: resolves `--llm` (alias / registry key / raw HF id) into an LLMSpec.
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from ..paths import CONFIGS
from ..utils import load_yaml


@dataclass
class LLMSpec:
    key: str
    hf_id: str
    params: str = "?"
    mode: str = "full"                     # full | lora | qlora
    min_vram_gb: float = 0
    gated: bool = False
    note: str = ""
    lora: dict = field(default_factory=dict)

    @property
    def tag(self) -> str:
        """Filesystem-safe name used in run directories."""
        return self.key.replace("/", "_")


@lru_cache(maxsize=1)
def _registry() -> dict:
    return load_yaml(CONFIGS / "llms.yaml")


def list_llms() -> list[LLMSpec]:
    reg = _registry()
    return [LLMSpec(key=k, lora=reg["lora_defaults"], **v) for k, v in reg["models"].items()]


def resolve_llm(name: str | None = None, mode: str | None = None) -> LLMSpec:
    """`name` may be an alias ('1.5b'), a registry key ('qwen2.5-1.5b') or any HF causal-LM id.
    Unregistered HF ids are allowed but unverified (defaults to LoRA)."""
    reg = _registry()
    name = name or reg["default"]
    name = reg["aliases"].get(name, name)
    if name in reg["models"]:
        spec = LLMSpec(key=name, lora=dict(reg["lora_defaults"]), **reg["models"][name])
    else:
        spec = LLMSpec(key=name, hf_id=name, mode="lora", lora=dict(reg["lora_defaults"]),
                       note="unregistered model: compatibility not verified")
    if mode and mode != "auto":
        if mode not in ("full", "lora", "qlora"):
            raise ValueError(f"mode must be full|lora|qlora, got {mode}")
        spec.mode = mode
    return spec
