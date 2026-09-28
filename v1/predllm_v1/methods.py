"""
Method presets = the rows of Table III. Every LLM method is ONE pipeline with three switches:

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

from typing import Any, Optional, List, Dict, Union

class MethodSpec:
    def as_dict(self):
        pass

# Placeholder for constant / export
GROUPS: Any = None

def resolve_methods(arg):
    pass
