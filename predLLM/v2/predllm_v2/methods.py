"""v2 presets = config overrides on top of v2/configs/default.yaml. Presets that differ only in
decoding / querying / curation share one fine-tuned model through the model cache."""
from __future__ import annotations

PRESETS: dict[str, list[str]] = {
    "pred_llm_v2": [],                                                        # everything on
    "v2_no_curation": ["curation.enabled=false"],                             # - N5
    "v2_R1": ["query.permutations=1"],                                        # - N4 order marginalisation
    "v2_argmax": ["query.label_decode=argmax"],                               # label decision rule
    "v2_plain_loss": ["loss.target_weight=1.0", "loss.scaffold_weight=1.0"],  # - N3
    "v2_encoded": ["serialization=encoded"],                                  # - N1 (paper's encoding)
}
GROUPS = {"v2": ["pred_llm_v2"], "ablation": list(PRESETS),
          "compare": ["original", "pred_llm_v2", "ctgan", "tvae"]}


def resolve_methods(arg: str) -> list[str]:
    out = []
    for item in arg.split(","):
        out += GROUPS.get(item.strip(), [item.strip()])
    for m in out:
        if m not in PRESETS and m not in ("original", "ctgan", "tvae", "copulagan"):
            raise KeyError(f"unknown v2 method '{m}'. presets: {list(PRESETS)}, groups: {list(GROUPS)}")
    return list(dict.fromkeys(out))
