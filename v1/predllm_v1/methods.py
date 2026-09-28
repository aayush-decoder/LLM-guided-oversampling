"""Method presets = the rows of Table III. Every LLM method is ONE pipeline with three switches:

  permutation  'x'  : shuffle features only, target fixed last          (Pred-LLM, Sec. III-B1b)
               'xy' : shuffle features AND target                        (GReaT / TapTap)
  sampling     'feature' : start token "X_i is v", X_i cycled uniformly, v ~ p(X_i)  (Sec. III-B2)
               'target'  : start token "target is y", y ~ p(Y)          (class-conditional)
  query        'llm'        : prompt "X_1 is v_1, ..., X_M is v_M" -> LLM completes y   (Sec. III-B3)
               'classifier' : XGBoost trained on D_real predicts y (TapTap)
               'none'       : keep the y generated together with x (GReaT)
  augment_original : also fine-tune on the rows in original column order (Alg. 1 line 7) - needed so
                     the original-order query prompt is in-distribution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    permutation: str
    sampling: str
    query: str
    augment_original: bool
    paper_row: str = ""

    def as_dict(self):
        return asdict(self)


PRESETS: dict[str, MethodSpec] = {m.name: m for m in [
    MethodSpec("great", "xy", "target", "none", False, "Great (Table III row 1)"),
    MethodSpec("taptap", "xy", "target", "classifier", False, "TapTap (Table III row 2)"),
    MethodSpec("abl_x_target_none", "x", "target", "none", True, "permute x / target / none"),
    MethodSpec("abl_xy_feature_none", "xy", "feature", "none", True, "permute xy / each feature / none"),
    MethodSpec("abl_xy_target_llm", "xy", "target", "llm", True, "permute xy / target / LLM"),
    MethodSpec("abl_xy_feature_llm", "xy", "feature", "llm", True, "permute xy / each feature / LLM"),
    MethodSpec("abl_x_target_llm", "x", "target", "llm", True, "permute x / target / LLM"),
    MethodSpec("pred_llm", "x", "feature", "llm", True, "Pred-LLM (Table III last row)"),
]}

GROUPS = {
    "llm": ["great", "taptap", "pred_llm"],
    "ablation": list(PRESETS),
    "sdv": ["ctgan", "tvae", "copulagan"],
    "paper": ["original", "copulagan", "ctgan", "tvae", "great", "taptap", "pred_llm"],
}


def resolve_methods(arg: str) -> list[str]:
    out = []
    for item in arg.split(","):
        out += GROUPS.get(item.strip(), [item.strip()])
    for m in out:
        if m not in PRESETS and m not in ("original", "ctgan", "tvae", "copulagan"):
            raise KeyError(f"unknown method '{m}'. presets: {list(PRESETS)}, groups: {list(GROUPS)}")
    return list(dict.fromkeys(out))
