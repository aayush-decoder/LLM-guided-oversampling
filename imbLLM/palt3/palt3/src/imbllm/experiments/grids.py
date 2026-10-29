"""Named experiments. Each maps (datasets, seeds, llm settings) -> WorkItems.

    baselines     references + classical (+ CTGAN/TVAE/TabDDPM/ARF if the
                  optional backends are installed)                          Table II cols
    paper         ImbLLM as published                                        Table II row
    llm_baselines GReaT / TapTap / Pred-LLM style                            Table II cols
    ablation      the 8 rows of Table III + ImbLLM-inter (Table IV)
    v2_compare    v1_exact vs recommended vs full                           headline
    v2_flags      v1_exact, v1_exact + ONE v2 flag (each), recommended,
                  recommended - ONE flag (each)                              attribution
    interp        the five interpolation geometries, all else v1             FIX B1
    tau           tau in {0, .25, .5, .75, 1} on `recommended`               FIX B2
    temperature   T in {0.5, 0.7, 1.0, 1.3}                                  frontier
    gibbs         SIR-Gibbs sweeps 0 / 1 / 2 + order sensitivity             FIX B4
    prior2x2      {pretrained, random init} x {named, anonymised columns}    P8
    order         order sensitivity: permuted-order vs FIXED-order fine-tune,
                  each with the real-row reference                          review 3.3
    curation      {ImbLLM, v2, SMOTE-NC} x {raw, CLLM-curated}              review G5
    qsweep        imbalance ratio q in {.1,.2,.3,.4,.5}                      Fig. 6
    rsweep        interpolation ratio r in {0,.25,.5,.75,1}                   Fig. 5
    smoke         tiny-random backbone, few epochs, every code path          CI

Sampling-only sweeps (tau, temperature, gibbs, condition) re-use ONE cached
fine-tune per (dataset, seed); only corpus-changing arms retrain.
"""

from __future__ import annotations

from ..core.baselines import classical, deep
from ..core.evaluate.downstream import REFERENCES
from ..methods.llm_baselines import NAMES as LLM_BASELINES
from ..methods.v1_paper.method import ABLATION
from ..methods.v2_corrected.method import V1_VALUES, V2_VALUES
from .runner import WorkItem

# default anchor for paired tests in each report
ANCHORS = {"paper": None, "ablation": "full_imbllm", "v2_compare": "v1_exact",
           "v2_flags": "v1_exact", "interp": "interp_random", "tau": "tau_1.0",
           "gibbs": "gibbs_0", "prior2x2": "pretrained_named", "baselines": "imbalance",
           "llm_baselines": "great_style", "temperature": "T_0.7", "qsweep": None,
           "rsweep": "r_1.0", "order": "order_fix_y", "curation": "imbllm_paper",
           "smoke": None}


def _items(exp, datasets, seeds, arms, protocol, q):
    return [WorkItem(exp, d, s, name, kind, cfg, protocol, q)
            for d in datasets for s in seeds for name, kind, cfg in arms]


def grid(name: str, datasets: list[str], seeds: list[int], llm: dict | None = None, *,
         protocol: str = "clean", q: float = 0.2, include_deep: bool | None = None
         ) -> list[WorkItem]:
    llm = dict(llm or {})
    v2 = lambda **kw: {**llm, **kw}          # noqa: E731
    arms: list

    if name == "baselines":
        arms = [(r, "reference", {}) for r in REFERENCES]
        arms += [(c, "classical", {"scale": True}) for c in classical.REGISTRY]
        arms += [(c, "deep", {}) for c in deep.REGISTRY
                 if (include_deep if include_deep is not None else deep.available(c))]
    elif name == "paper":
        arms = [("imbllm_paper", "v1", dict(llm))]
    elif name == "llm_baselines":
        arms = [(b, "llm_base", {**llm, "baseline": b}) for b in LLM_BASELINES]
    elif name == "ablation":
        arms = [(tag, "v1", {**llm, "condition": c, "permutation": p, "finetune_on": f})
                for tag, (c, p, f) in ABLATION.items()]
    elif name == "v2_compare":
        arms = [(p, "v2", v2(preset=p)) for p in ("v1_exact", "recommended", "full")]
    elif name == "v2_flags":
        arms = [("v1_exact", "v2", v2(preset="v1_exact")),
                ("recommended", "v2", v2(preset="recommended"))]
        for flag in V1_VALUES:
            arms.append((f"v1_plus_{flag}", "v2", v2(preset="v1_exact", **{flag: V2_VALUES[flag]})))
            if flag != "gibbs_sweeps":
                arms.append((f"rec_minus_{flag}", "v2",
                             v2(preset="recommended", **{flag: V1_VALUES[flag]})))
    elif name == "interp":
        from ..methods.v2_corrected.interpolate import MODES
        arms = [(f"interp_{m}", "v2", v2(preset="v1_exact", interpolation_mode=m))
                for m in MODES]
    elif name == "tau":
        arms = [(f"tau_{t}", "v2", v2(preset="recommended", tau=t))
                for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    elif name == "temperature":
        arms = [(f"T_{t}", "v2", v2(preset="recommended", temperature=t))
                for t in (0.5, 0.7, 1.0, 1.3)]
    elif name == "gibbs":
        arms = [(f"gibbs_{k}", "v2", v2(preset="recommended", gibbs_sweeps=k))
                for k in (0, 1, 2)]
    elif name == "prior2x2":
        arms = [(f"{i}_{'anon' if a else 'named'}", "v1", {**llm, "init": i,
                                                          "anonymize_columns": a})
                for i in ("pretrained", "random") for a in (False, True)]
    elif name == "order":
        # the fixed-order ("identity") fine-tune is the reference: it is trained on ONE
        # factorisation, so it SHOULD be order-sensitive. A permuted-order model that is
        # no better than it, or no better than its own real-row spread, has not learned
        # an order-free joint. Sampling-only knobs are held at `recommended`.
        arms = [(f"order_{p}", "v2", v2(preset="recommended", permutation=p,
                                         measure_order_sensitivity=True))
                for p in ("fix_y", "permute_xy", "identity")]
    elif name == "curation":
        arms = []
        for base, kind, cfg in (("imbllm_paper", "v1", dict(llm)),
                                ("v2_recommended", "v2", v2(preset="recommended")),
                                ("smote_nc", "classical", {"baseline": "smote_nc"})):
            arms.append((base, kind, cfg))
            arms.append((f"{base}+cllm", kind, {**cfg, "curate": "cllm"}))
    elif name == "qsweep":
        base = (("imbalance", "reference", {"reference": "imbalance"}),
                ("smote_nc", "classical", {"baseline": "smote_nc"}),
                ("imbllm", "v1", dict(llm)))
        return [WorkItem(name, d, s, f"{n}_q{qq}", k, c, protocol, qq)
                for qq in (0.1, 0.2, 0.3, 0.4, 0.5) for d in datasets for s in seeds
                for n, k, c in base]
    elif name == "rsweep":
        arms = [(f"r_{r}", "v1", {**llm, "interpolation_ratio": r})
                for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
    elif name == "smoke":
        smoke = {"model": "tiny-random", "epochs": 60, "lr": 2e-3, "effective_batch": 16,
                 "interpolation_ratio": 0.3,
                 "max_rounds": 2, "oversample_attempts": 2.0, **llm}
        arms = [("imbalance", "reference", {}), ("threshold_only", "reference", {}),
                ("prior_shift", "reference", {}), ("smote_nc", "classical", {}),
                ("imbllm_paper", "v1", dict(smoke)),
                ("v2_recommended", "v2", {**smoke, "preset": "recommended"}),
                ("v2_full", "v2", {**smoke, "preset": "full", "gibbs_max_rows": 16}),
                ("taptap_style", "llm_base", {**smoke, "baseline": "taptap_style"})]
    else:
        raise KeyError(f"unknown experiment {name!r}; have {sorted(ANCHORS)}")
    return _items(name, datasets, seeds, arms, protocol, q)


EXPERIMENTS = tuple(ANCHORS)
