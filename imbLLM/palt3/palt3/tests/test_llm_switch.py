"""The model switch and the hardware planner (no torch needed)."""

from __future__ import annotations

import pytest

from imbllm.core.llm import hardware, zoo


def P(name):
    return hardware.load_profile(name)


def test_only_compatible_models_are_listed():
    for bad in ("bert-base", "t5-small", "gpt-4o", "Qwen2.5-7B-Instruct", "openelm-270m"):
        with pytest.raises(ValueError):
            zoo.get(bad)
    for m in zoo.MODELS.values():
        assert "instruct" not in m.hf_id.lower() and "chat" not in m.hf_id.lower()


@pytest.mark.parametrize("model,profile,mode", [
    ("distilgpt2", "rtx4050_6gb", "full"),
    ("qwen2.5-1.5b", "rtx4050_6gb", "lora"),
    ("qwen2.5-3b", "rtx4050_6gb", "qlora"),
    ("qwen2.5-7b", "t4_16gb", "qlora"),
    ("qwen3-8b", "h100_80gb", "lora"),
    ("qwen2.5-3b", "h100_80gb", "full"),
    ("distilgpt2", "cpu", "full"),
])
def test_planner_picks_the_expected_mode(model, profile, mode):
    pl = hardware.plan(zoo.get(model), P(profile))
    assert pl.fits and pl.mode == mode, pl.reason


def test_planner_refuses_instead_of_ooming():
    assert not hardware.plan(zoo.get("qwen3-8b"), P("rtx4050_6gb")).fits
    assert not hardware.plan(zoo.get("qwen2.5-7b"), P("p100_16gb")).fits   # no 4-bit on Pascal
    assert not hardware.plan(zoo.get("qwen2.5-1.5b"), P("cpu")).fits


def test_precision_follows_the_device():
    assert hardware.plan(zoo.get("distilgpt2"), P("rtx4050_6gb")).precision == "bf16"
    assert hardware.plan(zoo.get("distilgpt2"), P("t4_16gb")).precision == "fp16"
    assert hardware.plan(zoo.get("distilgpt2"), P("cpu")).precision == "fp32"


def test_effective_batch_is_the_paper_s_32_everywhere():
    for m, prof in (("distilgpt2", "rtx4050_6gb"), ("qwen2.5-3b", "rtx4050_6gb"),
                    ("qwen3-8b", "h100_80gb")):
        pl = hardware.plan(zoo.get(m), P(prof))
        assert pl.micro_batch * pl.grad_accum == 32


def test_gated_models_need_a_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(PermissionError):
        zoo.check_access(zoo.get("llama3.1-8b"))
