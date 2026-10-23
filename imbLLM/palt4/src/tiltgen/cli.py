"""`tilt` -- the single command line of palt4 (TILT).  `tilt <cmd> -h` for flags.

    DATA        tilt data list | fetch | verify | inspect
    FIGURES     tilt viz data | synth | report
    LLM SWITCH  tilt models | tilt plan | tilt pull | tilt hw | tilt where-llm
    TRAIN       tilt train   one method on datasets x seeds (fit + generate + evaluate)
    INFER       tilt infer   fit (or load cached) and print generated rows next to real ones
    EXPERIMENTS tilt run <experiment>      named grids (paper tables, ablations, sweeps)
    REPORT      tilt report <experiment>   table + Friedman/Nemenyi + Holm-corrected paired tests
    AUDITS      tilt audit math | paper | data | theory | all
                tilt audit contamination --model qwen3-1.7b   (loads the backbone)
    ORACLE      tilt oracle                no-LLM study: does tilting help a PERFECT generator?
    SMOKE       tilt smoke                 every code path on an offline tiny model
    HOUSEKEEP   tilt clean --models | --runs <experiment>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import config as C
from . import paths
from .core.baselines import classical
from .core.data import fetch, protocol
from .core.llm import hardware
from .experiments import grids, report


# --------------------------------------------------------------------------- helpers

def _llm_overrides(a) -> dict:
    """LLM settings from --config YAML, then explicit flags, then --set k=v."""
    cfg = C.experiment_config(getattr(a, "config", None))
    for k in ("datasets", "seeds", "protocol", "q"):
        cfg.pop(k, None)
    for k in ("model", "hardware", "epochs", "finetune_mode", "temperature", "lr"):
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    for kv in getattr(a, "set", None) or []:
        k, _, v = kv.partition("=")
        import yaml
        cfg[k.strip()] = yaml.safe_load(v)
    return cfg


def _yaml_run_defaults(a) -> None:
    y = C.experiment_config(getattr(a, "config", None))
    if getattr(a, "datasets", None) is None:
        a.datasets = y.get("datasets", "natural_fast")
    if getattr(a, "seeds", None) is None:
        a.seeds = y.get("seeds", [0])
    if getattr(a, "protocol", None) is None:
        a.protocol = y.get("protocol", "clean")
    if getattr(a, "q", None) is None:
        a.q = y.get("q", 0.2)


def _common_run(p):
    p.add_argument("--datasets", default=None, help="names or groups: paper, predllm, fast, "
                   "open, all, smoke (or comma list)")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--config", default=None, help="configs/experiments/<name>.yaml or a path")
    p.add_argument("--model", default=None, help="backbone key; see `tilt models`")
    p.add_argument("--hardware", default=None, help="auto | configs/hardware profile")
    p.add_argument("--finetune-mode", dest="finetune_mode", default=None,
                   choices=[None, "auto", "full", "lora", "qlora"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE",
                   help="any method config field, e.g. --set tau=0.5 gibbs_sweeps=1")
    p.add_argument("--protocol", default=None, choices=[None, "clean", "paper"])
    p.add_argument("--q", type=float, default=None, help="imbalance ratio (paper: 0.2)")
    p.add_argument("--resume", action="store_true", help="skip finished identical runs")
    p.add_argument("--shard", default=None, help="i/N: run every N-th item (multi-GPU)")
    p.add_argument("--strict", action="store_true", help="stop at the first failure")
    p.add_argument("--dry-run", action="store_true", help="list work items and exit")


# --------------------------------------------------------------------------- data

def cmd_data(a) -> int:
    from .core.data import registry
    if a.sub == "list":
        print(f"{'dataset':<12}{'task':<11}{'source':<9}{'N':>7}{'M':>4}{'K':>4}  "
              f"{'on disk':<8}groups")
        for n, s in registry.DATASETS.items():
            e = s.expected
            groups = ",".join(g for g, m in registry.GROUPS.items() if n in m and g != "all")
            print(f"{n:<12}{s.task:<11}{s.source['type']:<9}{e.get('N', ''):>7}"
                  f"{e.get('M', ''):>4}{e.get('K', '') or '':>4}  "
                  f"{'yes' if fetch.raw_path(n).exists() else '-':<8}{groups}"
                  f"{'' if s.verified_source else '  [SOURCE UNVERIFIED]'}")
        return 0
    names = registry.resolve(a.datasets)
    if a.sub == "fetch":
        bad = 0
        for n in names:
            try:
                p = fetch.fetch(n, force=a.force, import_dir=a.from_dir)
                print(f"  {n:<12} ok   {p}")
            except Exception as exc:
                bad += 1
                print(f"  {n:<12} FAIL {exc}", file=sys.stderr)
        return 1 if bad else 0
    if a.sub == "verify":
        rows = [protocol.verify(n) for n in names]
        for r in rows:
            if r["status"] == "error":
                print(f"  {r['dataset']:<12} ERROR    {r['error']}")
            else:
                print(f"  {r['dataset']:<12} {r['status']:<9}actual={r['actual']}  "
                      f"delta={r['delta']}")
        paths.DATA.mkdir(parents=True, exist_ok=True)
        (paths.DATA / "verify_report.json").write_text(json.dumps(rows, indent=2, default=str))
        print(f"\nwrote {paths.DATA / 'verify_report.json'}. A mismatch is a finding to "
              f"record, never a reason to reshape data.")
        return 0
    if a.sub == "inspect":
        s = protocol.prepare(a.dataset, seed=a.seed, q=a.q, protocol=a.protocol)
        print(json.dumps({**s.summary(), "log": s.log}, indent=2, default=str))
        print(f"\ncontinuous ({s.schema.m_con}): {s.schema.continuous}")
        print(f"categorical ({s.schema.m_cat}): {s.schema.categorical}")
        import pandas as pd
        pd.set_option("display.width", 200, "display.max_columns", 40)
        print("\nD_minor (what the generator learns from):")
        print(s.train_minor[s.schema.features].head(8).to_string(index=False))
        if a.export:
            print(f"\nexported -> {protocol.export_splits(s)}")
        return 0
    return 2


# --------------------------------------------------------------------------- viz

def cmd_viz(a) -> int:
    from .core import viz
    if a.sub == "data":
        from .core.data import registry
        for n in registry.resolve(a.datasets):
            for p in viz.data_report(n, seed=a.seed, q=a.q, protocol=a.protocol):
                print(f"  {p}")
    elif a.sub == "synth":
        for p in viz.synth_report(Path(a.run)):
            print(f"  {p}")
    elif a.sub == "report":
        from .experiments import report
        p = viz.report_figure(report.build(a.experiment, a.metric))
        print(f"  {p}")
    return 0


# --------------------------------------------------------------------------- LLM switch

def cmd_models(a) -> int:
    from .core.llm import zoo
    profiles = [hardware.load_profile(p.stem) for p in sorted(paths.HARDWARE_CONFIGS.glob("*.yaml"))]
    hdr = "".join(f"{p.name[:13]:>14}" for p in profiles)
    print(f"{'key':<14}{'params':>7}  {'family':<8}{hdr}   (mode chosen per device)")
    for m in zoo.MODELS.values():
        cells = ""
        for hw in profiles:
            pl = hardware.plan(m, hw, seq_len=a.seq_len)
            cells += f"{(pl.mode + '/' + pl.precision) if pl.fits else '-- no --':>14}"
        flag = " [GATED]" if m.gated else ""
        print(f"{m.key:<14}{m.params_b:>6.2f}B  {m.family:<8}{cells}{flag}")
    print("\nOnly decoder-only BASE checkpoints are listed; rejected on purpose:")
    for k, why in zoo.INCOMPATIBLE.items():
        print(f"  {k:<10} {why}")
    print("\n`tilt plan --model X --hardware Y` shows micro-batch / accumulation / VRAM estimate.")
    return 0


def cmd_plan(a) -> int:
    from .core.llm import zoo
    hw = hardware.get_hardware(a.hardware)
    pl = hardware.plan(zoo.get(a.model), hw, mode=a.mode, seq_len=a.seq_len)
    print(json.dumps({"hardware": hw.as_dict(), "plan": pl.as_dict()}, indent=2))
    return 0 if pl.fits else 1


def cmd_pull(a) -> int:
    """Download backbone weights + tokenizer into the HF cache ahead of time.
    (Otherwise they download automatically on the first `tilt train/run/infer`.)"""
    from .core.llm import zoo
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print('needs the llm extra: pip install -e ".[llm]"')
        return 1
    for key in a.model:
        spec = zoo.get(key)
        if spec.family == "tiny":
            print(f"  {key}: built locally, nothing to download")
            continue
        zoo.check_access(spec)
        p = snapshot_download(spec.hf_id, allow_patterns=["*.json", "*.safetensors",
                                                         "*.model", "*.txt", "*.tiktoken"])
        print(f"  {key:<14} -> {p}")
    return 0


def cmd_hw(a) -> int:
    from .core.llm import hardware
    print("detected:", json.dumps(hardware.detect().as_dict(), indent=2))
    print("\nprofiles (configs/hardware):")
    for p in sorted(paths.HARDWARE_CONFIGS.glob("*.yaml")):
        h = hardware.load_profile(p.stem)
        print(f"  {p.stem:<14} {h.device:<5} {h.vram_gb:>5} GB  bf16={h.bf16!s:<5} "
              f"qlora={h.qlora!s:<5} full<= {h.full_ft_max_b}B")
    return 0


WHERE_LLM = """The LLM is touched in exactly one module: src/tiltgen/core/llm/engine.py
  LLMEngine.finetune(texts)          causal-LM fine-tune on serialised rows
  LLMEngine.complete(prompts)        sample row continuations
  LLMEngine.score(ctx, suffixes)     sum log p(suffix | context)

Call sites (the only places a method uses the LLM):
  methods/pipeline.py       LLMOversampler.fit       -> finetune   (all LLM methods)
  methods/pipeline.py       LLMOversampler.generate  -> complete   (all LLM methods)
  methods/v2_corrected/gibbs.py   sir_gibbs          -> score      (v2, gibbs_sweeps > 0)
  methods/v2_corrected/gibbs.py   order_sensitivity  -> score      (v2 diagnostic)
  methods/v2_corrected/method.py  measure            -> complete   (condition_y twin for
                                                                    the entropy decomposition)
  methods/llm_baselines.py  _llm_verifier            -> score      (predllm_style label query)
  audits/contamination.py   probe                    -> complete   (greedy; pre-trained
                                                                    backbone, no fine-tune)
  methods/tilt/method.py    _sir                     -> complete   (pool from the v2 generator)
  methods/tilt/smc.py       twisted_smc              -> complete   (one FIELD per call, per
                                                                    particle)
  methods/tilt/method.py    _contrast                -> score      (utility `contrast`: row
                                                                    log-ratio y_maj vs y_min)

Model choice: core/llm/zoo.py.  Device plan: core/llm/hardware.py.
Nothing else imports torch / transformers."""


def cmd_where_llm(a) -> int:
    print(WHERE_LLM)
    if a.grep:
        print("\ngrep of the source tree:")
        root = Path(__file__).parent
        for f in sorted(root.rglob("*.py")):
            if f.name == "cli.py":
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                code = line.strip()
                if any(t in code for t in ("engine.finetune(", ".complete(", "engine.score(")) \
                        and not code.startswith(("def ", "#")) and "<-" not in code \
                        and "=" in code:
                    print(f"  {f.relative_to(root.parent)}:{i}: {line.strip()}")
    return 0


# --------------------------------------------------------------------------- train / run

def _method_item(method: str, preset: str | None, llm: dict):
    from .core.baselines import deep
    from .core.evaluate.downstream import REFERENCES
    from .methods.llm_baselines import NAMES
    if method in REFERENCES:
        return method, "reference", {}
    if method in classical.REGISTRY:
        return method, "classical", {}
    if method in deep.REGISTRY:
        return method, "deep", {}
    if method in NAMES:
        return method, "llm_base", {**llm, "baseline": method}
    if method == "v1":
        return "v1_" + llm.get("model", "distilgpt2"), "v1", llm
    if method == "v2":
        p = preset or "recommended"
        return f"v2_{p}", "v2", {**llm, "preset": p}
    if method == "tilt":
        p = preset or "tilt_sir"
        return p, "tilt", {**llm, "preset": p}
    raise SystemExit(f"unknown --method {method!r}")


def _execute(items, a) -> int:
    from .core.runs import shard
    from .experiments import runner
    items = shard(items, a.shard)
    if a.dry_run:
        for it in items:
            print(f"  {it.experiment}/{it.dataset}/{it.method}/seed{it.seed}  kind={it.kind} "
                  f"q={it.q} cfg={json.dumps(it.config, default=str)[:120]}")
        print(f"{len(items)} items")
        return 0
    res = runner.run_all(items, resume=a.resume, strict=a.strict)
    return 1 if res["failed"] else 0


def cmd_train(a) -> int:
    from .core.data import registry
    from .experiments.runner import WorkItem
    _yaml_run_defaults(a)
    name, kind, cfg = _method_item(a.method, a.preset, _llm_overrides(a))
    items = [WorkItem(a.experiment, d, s, name, kind, cfg, a.protocol, a.q)
             for d in registry.resolve(a.datasets) for s in a.seeds]
    return _execute(items, a)


def cmd_run(a) -> int:
    from .core.data import registry
    from .experiments import grids
    _yaml_run_defaults(a)
    items = grids.grid(a.experiment, registry.resolve(a.datasets), a.seeds, _llm_overrides(a),
                       protocol=a.protocol, q=a.q)
    return _execute(items, a)


def cmd_infer(a) -> int:
    import pandas as pd

    from .core.data.protocol import prepare
    from .experiments.runner import WorkItem, build_llm
    _yaml_run_defaults(a)
    name, kind, cfg = _method_item(a.method, a.preset, _llm_overrides(a))
    if kind not in ("v1", "v2", "llm_base", "tilt"):
        raise SystemExit("infer needs an LLM method: v1 | v2 | great_style | taptap_style | "
                         "predllm_style")
    s = prepare(a.dataset, seed=a.seed, q=a.q, protocol=a.protocol)
    gen = build_llm(WorkItem("infer", a.dataset, a.seed, name, kind, cfg))
    gen.fit(s, seed=a.seed)
    syn = gen.generate(s, n=a.n, seed=a.seed)
    pd.set_option("display.width", 220, "display.max_columns", 60)
    print(f"\n=== REAL minority rows (D_minor, {len(s.train_minor)} rows) ===")
    print(s.train_minor[s.schema.features].head(a.n).to_string(index=False))
    print(f"\n=== GENERATED minority rows ({len(syn)}) ===")
    print(syn.head(a.n).to_string(index=False) if len(syn) else "(nothing parseable)")
    print("\n=== parse stats ===")
    print(json.dumps(gen.parse_stats.as_dict(), indent=2))
    print("\n=== prompts / plan / cache ===")
    print(json.dumps({k: gen.diagnostics.get(k) for k in ("prompts", "plan", "model_cache",
                                                         "token_lengths")}, indent=2, default=str))
    gen.release()
    return 0


def cmd_report(a) -> int:
    from .experiments import report
    res = report.build(a.experiment, a.metric, a.anchor, a.include_original)
    print(report.render(res))
    if "error" not in res:
        print(f"\nwrote {report.write(res)}")
        if a.figure:
            from .core import viz
            print(f"figure {viz.report_figure(res)}")
    return 0


# --------------------------------------------------------------------------- audits / smoke

def cmd_audit(a) -> int:
    failed = 0
    if a.sub == "contamination":
        from .audits import contamination
        from .core.data import registry
        from .core.runs import write_json
        ds = registry.resolve(a.datasets or "adult,german,bank,compas")
        res = contamination.run(ds, a.model, trials=a.trials, hardware=a.hardware)
        print(f"== contamination probes on pre-trained {a.model} (greedy) ==")
        for r in res:
            if "error" in r:
                print(f"  {r['dataset']:<12} {r['error']}")
                continue
            print(f"  {r['dataset']:<12} header {r['header_chars_matched']:>3}/"
                  f"{r['header_chars_total']:<4} row-exact {r['row_exact_match']:.2f}  "
                  f"feature({r['feature_column']}) {r['feature_exact_match']:.2f} vs base "
                  f"{r['feature_baseline']:.2f}  first-cell {r['first_cell_match']:.2f} vs base "
                  f"{r['first_cell_baseline']:.2f}  -> "
                  f"{'MEMORISED?' if r['memorisation_flag'] else 'no evidence'}")
        out = paths.RUNS / "_audits"
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / f"contamination_{a.model}.json", res)
        print(f"\nwrote {out / f'contamination_{a.model}.json'}")
        return 0
    a.datasets = a.datasets or "all"
    if a.sub in ("theory", "all"):
        from .audits import theory_checks
        print("== theory (docs/THEORY.md propositions, numerically) ==")
        for name, ok, det in theory_checks.run_all():
            failed += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
            if a.verbose:
                print("        " + json.dumps(det, default=str)[:900])
    if a.sub in ("math", "all"):
        from .audits import math_checks
        print("== math (shipped code) ==")
        for name, ok, det in math_checks.run_all():
            failed += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
            if a.verbose:
                print("        " + json.dumps(det, default=str)[:900])
    if a.sub in ("paper", "all"):
        from .audits import paper_tables as pt
        print("== paper tables ==")
        ok, res = pt.check_transcription()
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  Table II transcription (every column mean == "
              f"printed average)")
        if not ok:
            for k, (m, p) in res.items():
                if abs(m - p) > 5e-5:
                    print(f"        {k}: computed {m} vs printed {p}")
        t = pt.omitted_tests()
        print(f"  Friedman over 7 oversamplers: chi2={t['friedman_chi2']:.3f} "
              f"p={t['friedman_p']:.5f}  CD={t['cd']}")
        print(f"  mean ranks: {t['mean_ranks']}")
        print(f"  not separable from best: {t['not_separable_from_best']}")
        for k, r in t["paired_vs_imbllm"].items():
            hp = "" if r["holm_p"] is None else f" holm={r['holm_p']:.4f}"
            print(f"    ImbLLM vs {k:<10} diff={r['mean_diff']:+.4f} wins={r['wins']}/10 "
                  f"p={r['wilcoxon_p']:.4f}{hp}")
        print(f"  Table III decomposition: {pt.ablation_decomposition()}")
    if a.sub in ("data", "all"):
        from .audits import data_checks
        print("== data ==")
        for r in data_checks.run(a.datasets):
            if r.get("status") in ("not downloaded", "error"):
                print(f"  {r['dataset']:<12} {r['status']} {r.get('error', '')}")
                continue
            leak = r["single_feature_leaks"]
            print(f"  {r['dataset']:<12} {r['status']:<9} dup(train,test)="
                  f"{r['train_test_duplicate_rows']:<4} leaks={leak or '-'} "
                  f"disjoint={r['protocol_disjoint']}")
    print(f"\n{'ALL PASS' if not failed else f'{failed} FAILED'}")
    return 1 if failed else 0


def cmd_oracle(a) -> int:
    """No-LLM study (experiments/oracle.py): exact p_min samples, tilted or not."""
    from .experiments import oracle
    df = oracle.run(n_mins=tuple(a.n_min), seeds=range(a.seeds), learners=tuple(a.learners),
                    budgets=tuple(a.budgets), d=a.dim, sep=a.sep, bimodal=not a.unimodal,
                    ir=a.ir)
    out = paths.RUNS / "oracle"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"ir{a.ir:g}_d{a.dim}_{'uni' if a.unimodal else 'bi'}"
    df.to_csv(out / f"results_{tag}.csv", index=False)
    summ = oracle.summarise(df)
    summ.to_csv(out / f"summary_{tag}.csv")
    print(summ.to_string())
    print(f"\nwrote {out / f'results_{tag}.csv'} and summary_{tag}.csv")
    return 0


def cmd_smoke(a) -> int:
    """Every code path, offline: tiny-random backbone, the synthetic dataset,
    references, SMOTE-NC, v1, v2 recommended, v2 full (SIR-Gibbs), TapTap-style."""
    from .core.data import registry
    from .core.runs import collect
    from .experiments import runner
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print('smoke needs the llm extra: pip install -e ".[llm]"')
        return 1
    ds = registry.resolve(a.datasets)
    items = grids.grid("smoke", ds, [0], {"hardware": a.hardware or "auto"})
    res = runner.run_all(items, strict=a.strict)
    df = collect("smoke")
    need = ["manifest.json", "metrics.json", "splits.json"]
    llm_need = need + ["diagnostics.json", "parse_stats.json", "train_log.json", "synthetic.csv"]
    missing = []
    for it in items:
        d = paths.RUNS / "smoke" / it.dataset / it.method / "seed0"
        for f in (llm_need if it.kind in ("v1", "v2", "llm_base", "tilt") else need):
            if not (d / f).exists():
                missing.append(str(d / f))
    print("\n" + report.render(report.build("smoke", "f1")))
    ok = not res["failed"] and not missing and len(df) == len(items)
    print(f"\nartefacts missing: {missing or 'none'}")
    print(f"SMOKE {'PASS' if ok else 'FAIL'} ({len(items)} runs)")
    return 0 if ok else 1


def cmd_clean(a) -> int:
    from .core.runs import ModelCache
    if a.models:
        print(f"removed {ModelCache().clear()} cached model(s)")
    for e in a.runs or []:
        p = paths.RUNS / e
        if p.exists():
            shutil.rmtree(p)
            print(f"removed {p}")
    return 0


# --------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    from .experiments.grids import EXPERIMENTS
    p = argparse.ArgumentParser("tilt", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="dataset registry / fetch / verify / inspect")
    dsub = d.add_subparsers(dest="sub", required=True)
    dsub.add_parser("list")
    f = dsub.add_parser("fetch")
    f.add_argument("--datasets", default="open")
    f.add_argument("--from", dest="from_dir", default=None,
                   help="copy <name>.csv from this folder first (e.g. ../palt/data/raw)")
    f.add_argument("--force", action="store_true")
    v = dsub.add_parser("verify")
    v.add_argument("--datasets", default="all")
    i = dsub.add_parser("inspect")
    i.add_argument("--dataset", required=True)
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--q", type=float, default=0.2)
    i.add_argument("--protocol", default="clean", choices=["clean", "paper"])
    i.add_argument("--export", action="store_true", help="write the splits as csv")
    d.set_defaults(func=cmd_data)

    z = sub.add_parser("viz", help="figures: data before/after cleaning, runs, reports")
    zsub = z.add_subparsers(dest="sub", required=True)
    zd = zsub.add_parser("data")
    zd.add_argument("--datasets", default="fast")
    zd.add_argument("--seed", type=int, default=0)
    zd.add_argument("--q", type=float, default=0.2)
    zd.add_argument("--protocol", default="clean", choices=["clean", "paper"])
    zs = zsub.add_parser("synth")
    zs.add_argument("--run", required=True, help="a run directory runs/<exp>/<ds>/<m>/seed<k>")
    zr = zsub.add_parser("report")
    zr.add_argument("experiment")
    zr.add_argument("--metric", default="f1")
    z.set_defaults(func=cmd_viz)

    m = sub.add_parser("models", help="compatible backbones x hardware profiles")
    m.add_argument("--seq-len", dest="seq_len", type=int, default=256)
    m.set_defaults(func=cmd_models)
    pl = sub.add_parser("plan", help="how one backbone runs on one device")
    pl.add_argument("--model", required=True)
    pl.add_argument("--hardware", default="auto")
    pl.add_argument("--mode", default="auto", choices=["auto", "full", "lora", "qlora"])
    pl.add_argument("--seq-len", dest="seq_len", type=int, default=256)
    pl.set_defaults(func=cmd_plan)
    pu = sub.add_parser("pull", help="pre-download backbone weights (else: on first use)")
    pu.add_argument("--model", nargs="+", required=True)
    pu.set_defaults(func=cmd_pull)
    sub.add_parser("hw", help="detected device + hardware profiles").set_defaults(func=cmd_hw)
    w = sub.add_parser("where-llm", help="every place the LLM is used")
    w.add_argument("--grep", action="store_true")
    w.set_defaults(func=cmd_where_llm)

    t = sub.add_parser("train", help="run ONE method on datasets x seeds")
    t.add_argument("--method", required=True, help="tilt | v1 | v2 | great_style | taptap_style | "
                   "predllm_style | smote_nc | smote | adasyn | ros | ctgan | tvae | original"
                   " | imbalance | threshold_only | prior_shift | scale_pos_weight")
    t.add_argument("--preset", default=None, help="v2: v1_exact | recommended | "
                   "geometry_only | full;  tilt: tilt_sir | tilt_smc | tilt_off | "
                   "tilt_influence | tilt_epistemic | tilt_aleatoric | tilt_random | "
                   "tilt_contrast | tilt_nogate")
    t.add_argument("--experiment", default="train", help="runs/<experiment>/...")
    _common_run(t)
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("run", help="a named experiment grid")
    r.add_argument("experiment", choices=EXPERIMENTS)
    _common_run(r)
    r.set_defaults(func=cmd_run)

    inf = sub.add_parser("infer", help="fit/load and print generated rows")
    inf.add_argument("--method", default="v2")
    inf.add_argument("--preset", default=None)
    inf.add_argument("--dataset", default="synthetic")
    inf.add_argument("--n", type=int, default=10)
    inf.add_argument("--seed", type=int, default=0)
    _common_run(inf)
    inf.set_defaults(func=cmd_infer)

    rp = sub.add_parser("report", help="aggregate an experiment")
    rp.add_argument("experiment")
    rp.add_argument("--metric", default="f1")
    rp.add_argument("--anchor", default=None)
    rp.add_argument("--include-original", action="store_true")
    rp.add_argument("--figure", action="store_true")
    rp.set_defaults(func=cmd_report)

    au = sub.add_parser("audit", help="math / paper / data checks; contamination probes")
    au.add_argument("sub", choices=["math", "paper", "data", "theory", "all", "contamination"],
                    help="`all` = math+paper+data+theory (no LLM); `contamination` loads --model")
    au.add_argument("--datasets", default=None,
                    help="default: all (data) / adult,german,bank,compas (contamination)")
    au.add_argument("--model", default="distilgpt2", help="contamination: backbone key")
    au.add_argument("--trials", type=int, default=25, help="contamination: prompts per probe")
    au.add_argument("--hardware", default="auto")
    au.add_argument("-v", "--verbose", action="store_true")
    au.set_defaults(func=cmd_audit)

    orc = sub.add_parser("oracle", help="no-LLM study: tilting a PERFECT generator (CPU)")
    orc.add_argument("--n-min", dest="n_min", type=int, nargs="+", default=[20, 50, 100])
    orc.add_argument("--seeds", type=int, default=20, help="number of seeds (0..k-1)")
    orc.add_argument("--learners", nargs="+", default=["lr", "xgb"], choices=["lr", "xgb"])
    orc.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    orc.add_argument("--ir", type=float, default=20.0, help="imbalance ratio maj/min")
    orc.add_argument("--dim", type=int, default=4)
    orc.add_argument("--sep", type=float, default=2.0)
    orc.add_argument("--unimodal", action="store_true", help="single-mode minority")
    orc.set_defaults(func=cmd_oracle)

    sm = sub.add_parser("smoke", help="offline end-to-end check of every code path")
    sm.add_argument("--datasets", default="smoke")
    sm.add_argument("--hardware", default=None)
    sm.add_argument("--strict", action="store_true")
    sm.set_defaults(func=cmd_smoke)

    cl = sub.add_parser("clean", help="delete cached models and/or experiment runs")
    cl.add_argument("--models", action="store_true")
    cl.add_argument("--runs", nargs="*", default=None)
    cl.set_defaults(func=cmd_clean)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
