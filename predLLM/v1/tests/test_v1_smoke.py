"""End-to-end smoke test (downloads distilgpt2, ~150 fine-tuning steps on iris)."""
import pytest

from predLLM.common.tabgen.paths import runs_root


@pytest.mark.smoke
def test_v1_smoke_end_to_end():
    from predLLM.v1.predllm_v1.smoke import run_smoke
    assert run_smoke(llm="distilgpt2", max_steps=150) == 0
    report = (runs_root("v1") / "_smoke" / "SMOKE_REPORT.md").read_text(encoding="utf-8")
    assert "OVERALL: PASS" in report
