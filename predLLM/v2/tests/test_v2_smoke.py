"""End-to-end v2 smoke test with the random tiny-gpt2 (a few MB): exercises fine-tuning, schema-constrained
decoding, the permutation-marginalised posterior, curation, metrics, plots and audits."""
import pytest

from predLLM.common.tabgen.paths import runs_root


@pytest.mark.smoke
def test_v2_smoke_end_to_end():
    from predLLM.v2.predllm_v2.smoke import run_smoke
    assert run_smoke(llm="tiny-gpt2", max_steps=20) == 0
    report = (runs_root("v2") / "_smoke" / "SMOKE_REPORT.md").read_text(encoding="utf-8")
    assert "OVERALL: PASS" in report
