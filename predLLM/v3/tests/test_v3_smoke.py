"""End-to-end v3 smoke test on iris using tiny-gpt2:
Exercises negative fine-tuning, outlier-bounded decoding, posterior querying, and curation.
"""
import pytest

from predLLM.common.tabgen.paths import runs_root


@pytest.mark.smoke
def test_v3_smoke_end_to_end():
    from predLLM.v3.predllm_v3.smoke import run_smoke
    assert run_smoke(llm="tiny-gpt2", max_steps=20) == 0
    report = (runs_root("v3") / "_smoke" / "SMOKE_REPORT.md").read_text(encoding="utf-8")
    assert "OVERALL: PASS" in report
