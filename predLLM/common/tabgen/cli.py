"""Shared CLI:  python -m tabgen <group> <command> [...]

  data   download | info | list
  viz    data | run
  audit  env | data | run | all-runs
  llm    list | check
  report --versions v1,v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .utils import save_json


def _datasets(arg):
    from .data.registry import resolve_datasets
    return resolve_datasets(arg)


def cmd_data(a):
    from .data.download import ManualDownloadRequired, fetch, is_available
    from .data.registry import get_spec
    if a.cmd == "list":
        for n in _datasets("all"):
            s = get_spec(n)
            print(f"{n:12s} {s.task:14s} {s.source['type']:7s} groups={s.group} available={is_available(n)}")
        return 0
    failed = []
    for n in _datasets(a.dataset):
        if a.cmd == "download":
            try:
                fetch(n, force=a.force)
            except ManualDownloadRequired as e:
                print(e)
                failed.append(n)
        elif a.cmd == "info":
            from .data.preprocess import make_split
            sp = make_split(n, seed=0)
            print(sp.summary())
    if failed:
        print(f"\nManual download needed for: {failed}  -> see docs/DATASETS.md")
    return 1 if failed else 0


def cmd_viz(a):
    if a.cmd == "data":
        from .viz.data_plots import plot_dataset
        for n in _datasets(a.dataset):
            out = plot_dataset(n, stages=a.stages.split(","))
            print(f"{n}: wrote {len(out)} files to {out[0].parent}")
    else:
        from .data.preprocess import make_split
        from .utils import load_json
        import pandas as pd
        for rd in _run_dirs(a.run):
            cfg = load_json(rd / "config.json")
            d = cfg["data"]
            sp = make_split(cfg["dataset"], cfg["seed"], d["test_size"], d["train_size"], d["fit_scaler_on"],
                            d.get("binarize"), d.get("max_rows"), d["decimals"])
            from .viz.synth_plots import plot_run
            print(rd, [p.name for p in plot_run(rd, sp, pd.read_csv(rd / "synthetic_encoded.csv"))])
    return 0


def _run_dirs(path) -> list[Path]:
    p = Path(path)
    if (p / "metrics.json").exists():
        return [p]
    return sorted(q.parent for q in p.rglob("metrics.json"))


def cmd_audit(a):
    from . import audit
    from .paths import REPORTS
    if a.cmd == "env":
        rep = audit.audit_env()
    elif a.cmd == "data":
        rep = [audit.audit_data(n) for n in _datasets(a.dataset)]
    else:
        rep = [audit.audit_run(rd) for rd in _run_dirs(a.run)]
    save_json(rep, REPORTS / "audits" / f"audit_{a.cmd}.json")
    statuses = [r["status"] for r in (rep if isinstance(rep, list) else [rep])]
    print(f"\nAudit {a.cmd}: " + ", ".join(f"{s}={statuses.count(s)}" for s in ("PASS", "WARN", "FAIL")))
    return 1 if "FAIL" in statuses else 0


def cmd_llm(a):
    from .llm.registry import list_llms, resolve_llm
    if a.cmd == "list":
        from .llm.loader import vram_gb
        g = vram_gb()
        print(f"GPU memory: {g:.1f} GB\n{'key':15s} {'params':7s} {'mode':6s} {'minGB':5s} fits gated hf_id")
        for s in list_llms():
            print(f"{s.key:15s} {s.params:7s} {s.mode:6s} {s.min_vram_gb:<5} {'yes ' if s.min_vram_gb <= g or s.min_vram_gb == 0 else 'no  '}"
                  f"{'yes  ' if s.gated else 'no   '} {s.hf_id}")
        return 0
    # check: load + one training step + one generation on the current device
    import torch
    from .llm.loader import load_llm
    from .llm.train_loop import make_collate, weighted_lm_loss
    from .utils import pick_device
    spec = resolve_llm(a.llm, a.mode)
    dev = pick_device(a.device)
    tok, model = load_llm(spec, dev)
    batch = make_collate(tok.pad_token_id)([{"input_ids": tok("age is 0.5, sex is 1, target is 0").input_ids}] * 4)
    batch = {k: v.to(dev) for k, v in batch.items()}
    loss = weighted_lm_loss(model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits,
                            batch["input_ids"], batch["weights"])
    loss.backward()
    out = model.generate(**tok("age is 0.5,", return_tensors="pt").to(dev), max_new_tokens=8, do_sample=True,
                         pad_token_id=tok.pad_token_id)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if dev.startswith("cuda") else 0
    print(f"OK {spec.key} ({spec.hf_id}, {spec.mode}) loss={loss.item():.3f} peak_mem={peak:.2f}GB "
          f"sample={tok.decode(out[0], skip_special_tokens=True)!r}")
    return 0


def cmd_report(a):
    from .eval.report import build_tables
    from .paths import ROOT
    versions = a.versions.split(",")
    out = Path(a.out) if a.out else ROOT / "reports" / "tables"
    written = build_tables(versions, out)
    for k, p in written.items():
        print(f"{k}: {p}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tabgen", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="group", required=True)
    d = sub.add_parser("data")
    d.add_argument("cmd", choices=["download", "info", "list"])
    d.add_argument("--dataset", default="all", help="name | a,b | all | paper | user | classification | regression")
    d.add_argument("--force", action="store_true")
    v = sub.add_parser("viz")
    v.add_argument("cmd", choices=["data", "run"])
    v.add_argument("--dataset", default="iris")
    v.add_argument("--stages", default="raw,clean,encoded")
    v.add_argument("--run", help="run dir (seed dir, experiment dir or whole runs/ tree)")
    au = sub.add_parser("audit")
    au.add_argument("cmd", choices=["env", "data", "run"])
    au.add_argument("--dataset", default="all")
    au.add_argument("--run", default="v1/runs")
    lm = sub.add_parser("llm")
    lm.add_argument("cmd", choices=["list", "check"])
    lm.add_argument("--llm", default=None)
    lm.add_argument("--mode", default="auto")
    lm.add_argument("--device", default="auto")
    rp = sub.add_parser("report")
    rp.add_argument("--versions", default="v1,v2")
    rp.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return {"data": cmd_data, "viz": cmd_viz, "audit": cmd_audit, "llm": cmd_llm, "report": cmd_report}[a.group](a)


if __name__ == "__main__":
    sys.exit(main())
