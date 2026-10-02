# Shared smoke harness: runs every given method once (seed 0) on a small dataset into
# <version>/runs/_smoke/, then checks that each required artifact exists and passes the run audit.
# Writes <version>/runs/_smoke/SMOKE_REPORT.md. Returns 0 when everything passes.
from __future__ import annotations


from .audit import REQUIRED_RUN_FILES, audit_run
from .baselines.sdv_models import SDV_METHODS
from .eval.report import build_tables
from .experiment import NON_LLM, exp_dir, run_experiment
from .paths import runs_root


def run_smoke(version: str, cfg: dict, make_generator, methods: list[str], llm_tag: str,
              dataset: str = "iris") -> int:
    cfg = {**cfg, "runs_subdir": "_smoke"}
    root = runs_root(version) / "_smoke"
    lines = [f"# {version} smoke report  (dataset={dataset}, llm={llm_tag})", ""]
    ok_all = True
    for m in methods:
        if m in SDV_METHODS:
            try:
                import sdv  # noqa: F401
            except ImportError:
                lines.append(f"- SKIP {m}: sdv not installed")
                continue
        try:
            summ = run_experiment(version, dataset, m, cfg, make_generator, llm_tag, [0], overwrite=True, plots=True)
            err = None
        except Exception as e:  # noqa: BLE001 - report and continue with the other methods
            summ, err = {}, f"{type(e).__name__}: {e}"
        rd = exp_dir(version, dataset, m, llm_tag, cfg) / "seed0"
        need = list(REQUIRED_RUN_FILES) + ["plots/marginals.png", "plots/tsne.png", "plots/dcr.png"]
        if m not in NON_LLM:
            need += ["model_ref.json", "train_log.jsonl"]
        missing = [f for f in need if not (rd / f).exists()]
        audit = audit_run(rd) if not missing and err is None else {"status": "FAIL"}
        ok = err is None and not missing and audit["status"] != "FAIL"
        ok_all &= ok
        lines.append(f"- {'PASS' if ok else 'FAIL'} **{m}**: {summ.get('metric')}={summ.get('mean')} "
                     f"(original {summ.get('original_mean')}) audit={audit['status']}"
                     + (f" missing={missing}" if missing else "") + (f" error={err}" if err else ""))
    tables = build_tables([version], root / "tables", subdir="_smoke")
    lines += ["", "Artifacts:", *[f"- {k}: {p}" for k, p in tables.items()]]
    lines += ["", f"OVERALL: {'PASS' if ok_all else 'FAIL'}"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "SMOKE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok_all else 1
