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

  palt4 / TILT (docs/EXPERIMENTS.md; anchor in brackets)
    thesis_main   references, SMOTE family, ImbLLM, v2, CLLM, TILT-SIR/SMC   [threshold_only]
    utility       TILT-SIR with each utility (fisher, influence, epistemic,
                  aleatoric, none, contrast)                                 [tilt_none]
    budget        KL budget eps in {0, .05, .1, .25, .5, 1, 2}               [eps_0]
    sampler       SIR pool x2/x4/x8 vs SMC particles 4/8/16                  [sir_x4]
    validity      validity floor / DCR gate on and off                       [tilt_sir]
    scale         Qwen3 Base ladder 0.6B .. 14B + Qwen2.5-32B (v2 vs TILT)   [-]
    scarcity      q in {.05, .1, .2, .5}: where does guidance matter most?   [-]
Every TILT arm except `contrast` reuses ONE cached fine-tune per (dataset, seed).

Sampling-only sweeps (tau, temperature, gibbs, condition) re-use ONE cached
fine-tune per (dataset, seed); only corpus-changing arms retrain.
"""

from __future__ import annotations

from ..core.baselines import deep
from ..core.baselines import classical
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
           "thesis_main": "threshold_only", "utility": "tilt_none", "budget": "eps_0",
           "sampler": "sir_x4", "validity": "tilt_sir", "scale": None, "scarcity": None,
           "smoke": None}

# the backbone ladder of the `scale` experiment (docs/COMPUTE.md)
SCALE_LADDER = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen2.5-32b")


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
    elif name in ("thesis_main", "utility", "budget", "sampler", "validity", "scale",
                  "scarcity"):
        return _tilt_grid(name, datasets, seeds, llm, protocol, q)
    elif name == "smoke":
        smoke = {"model": "tiny-random", "epochs": 60, "lr": 2e-3, "effective_batch": 16,
                 "interpolation_ratio": 0.3,
                 "max_rounds": 2, "oversample_attempts": 2.0, **llm}
        arms = [("imbalance", "reference", {}), ("threshold_only", "reference", {}),
                ("prior_shift", "reference", {}), ("smote_nc", "classical", {}),
                ("imbllm_paper", "v1", dict(smoke)),
                ("v2_recommended", "v2", {**smoke, "preset": "recommended"}),
                ("v2_full", "v2", {**smoke, "preset": "full", "gibbs_max_rows": 16}),
                ("taptap_style", "llm_base", {**smoke, "baseline": "taptap_style"}),
                ("tilt_sir", "tilt", {**smoke, "preset": "tilt_sir", "pool_factor": 2.0}),
                ("tilt_smc", "tilt", {**smoke, "preset": "tilt_smc", "pilot": 48,
                                      "particles": 4, "keep": 2})]
    else:
        raise KeyError(f"unknown experiment {name!r}; have {sorted(ANCHORS)}")
    return _items(name, datasets, seeds, arms, protocol, q)


def _tilt_grid(name, datasets, seeds, llm, protocol, q) -> list[WorkItem]:
    t = lambda preset, **kw: {**llm, "preset": preset, **kw}  # noqa: E731
    if name == "thesis_main":
        arms = [(r, "reference", {}) for r in REFERENCES]
        arms += [(c, "classical", {}) for c in ("smote_nc", "borderline_smote")]
        arms += [("imbllm_paper", "v1", dict(llm)),
                 ("v2_recommended", "v2", {**llm, "preset": "recommended"}),
                 ("v2_recommended+cllm", "v2", {**llm, "preset": "recommended",
                                                "curate": "cllm"}),
                 ("tilt_sir", "tilt", t("tilt_sir")),
                 ("tilt_smc", "tilt", t("tilt_smc"))]
    elif name == "utility":
        arms = [(f"tilt_{u}", "tilt", t("tilt_sir", utility=u))
                for u in ("fisher", "influence", "epistemic", "aleatoric", "none")]
        arms.append(("tilt_contrast", "tilt", t("tilt_contrast")))
    elif name == "budget":
        arms = [(f"eps_{e:g}", "tilt", t("tilt_sir", kl_budget=e))
                for e in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)]
    elif name == "sampler":
        arms = [(f"sir_x{f:g}", "tilt", t("tilt_sir", pool_factor=f)) for f in (2.0, 4.0, 8.0)]
        arms += [(f"smc_p{p}", "tilt", t("tilt_smc", particles=p, keep=max(1, p // 2)))
                 for p in (4, 8, 16)]
    elif name == "validity":
        arms = [("tilt_sir", "tilt", t("tilt_sir")),
                ("tilt_nogate", "tilt", t("tilt_nogate")),
                ("valid_0.2", "tilt", t("tilt_sir", valid_q=0.2)),
                ("no_dcr_gate", "tilt", t("tilt_sir", privacy_alpha=0.0))]
    elif name == "scale":
        arms = []
        for m in SCALE_LADDER:
            arms.append((f"v2@{m}", "v2", {**llm, "preset": "recommended", "model": m}))
            arms.append((f"tilt_sir@{m}", "tilt", t("tilt_sir", model=m)))
    elif name == "scarcity":
        base = (("threshold_only", "reference", {}), ("smote_nc", "classical", {}),
                ("v2", "v2", {**llm, "preset": "recommended"}),
                ("tilt_sir", "tilt", t("tilt_sir")))
        return [WorkItem(name, d, s, f"{n}_q{qq:g}", k, c, protocol, qq)
                for qq in (0.05, 0.1, 0.2, 0.5) for d in datasets for s in seeds
                for n, k, c in base]
    else:
        raise KeyError(name)
    return _items(name, datasets, seeds, arms, protocol, q)


EXPERIMENTS = tuple(ANCHORS)
