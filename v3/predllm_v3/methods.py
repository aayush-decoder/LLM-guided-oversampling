"""v3 presets and ablations for negative fine-tuning and outlier support generation."""
from __future__ import annotations

PRESETS: dict[str, list[str]] = {
    "pred_llm_v3": [],                                                      # full v3 pipeline
    "v3_no_negative": ["finetune.negative_weight=0.0"],                    # ablation: no negative unlikelihood
    "v3_no_outlier_range": ["range_mode=empirical"],                        # ablation: standard min-max range
    "v3_cfg_negative": ["decoding.negative_guidance=0.5"],                  # inference logit subtraction (CFG)
    "v3_no_curation": ["curation.enabled=false"],                           # ablation: no curation gates
    "v3_plain": [                                                           # baseline control
        "finetune.negative_weight=0.0",
        "range_mode=empirical",
        "curation.anomaly_gate=false",
    ],
}

GROUPS = {
    "v3": ["pred_llm_v3"],
    "ablation": list(PRESETS),
    "compare": ["original", "pred_llm_v3", "ctgan", "tvae"],
}


def resolve_methods(arg: str) -> list[str]:
    out = []
    for item in arg.split(","):
        out += GROUPS.get(item.strip(), [item.strip()])
    for m in out:
        if m not in PRESETS and m not in ("original", "ctgan", "tvae", "copulagan"):
            raise KeyError(f"unknown v3 method '{m}'. presets: {list(PRESETS)}, groups: {list(GROUPS)}")
    return list(dict.fromkeys(out))
