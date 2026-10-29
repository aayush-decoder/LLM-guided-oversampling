"""The paper's published numbers, verified cell by cell, and the tests it omits.

TABLE_II was re-read from the PDF (arXiv 2510.09783v2, p.7). palt2 had two
wrong cells (GReaT/german = 0.2760 -- TVAE's value -- instead of 0.2059, and
TapTap/german = 0.2059 instead of 0.2069) and had edited the TapTap average
check to .7176 to match its own typo. Here every column mean must reproduce
the printed average to 4 decimals, or the audit FAILS.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from ..core.evaluate.stats import critical_difference, holm

DATASETS = ("adult", "german", "fuel", "insurance", "gem_price", "housing", "bank", "car",
            "credit_card", "sick")

TABLE_II = {   # F1, Table II
    "Original":  (.6761, .5299, .9645, .9153, .9799, .9111, .3657, .9923, .9825, .9208),
    "Imbalance": (.4685, .2032, .9664, .9220, .9707, .8738, .1118, .9757, .9671, .7790),
    "AdaSyn":    (.3987, .4601, .9458, .3933, .2818, .4421, .2063, .7695, .0436, .1826),
    "SMOTE":     (.3987, .4595, .9406, .4263, .5931, .4450, .2064, .7689, .0701, .2057),
    "SMOTE-NC":  (.5586, .2706, .9662, .9253, .9739, .8801, .2322, .9684, .9725, .8270),
    "CTGAN":     (.5223, .4064, .9658, .8858, .9705, .8690, .3162, .9787, .9698, .8394),
    "TVAE":      (.4985, .2760, .9642, .9193, .9711, .8707, .2134, .9734, .9671, .8376),
    "GReaT":     (.6592, .2059, .9272, .7690, .9698, .8721, .2183, .5732, .9634, .6752),
    "TapTap":    (.4705, .2069, .9751, .9209, .9709, .8756, .0831, .9716, .9634, .7386),
    "Pred-LLM":  (.5071, .2363, .9662, .9182, .9733, .8826, .1352, .9831, .9692, .7740),
    "ImbLLM":    (.6543, .4419, .9820, .9250, .9755, .8750, .3441, .9820, .9758, .8459),
}
PRINTED_AVG = {"Original": .8238, "Imbalance": .7238, "AdaSyn": .4124, "SMOTE": .4514,
               "SMOTE-NC": .7575, "CTGAN": .7724, "TVAE": .7491, "GReaT": .6833,
               "TapTap": .7177, "Pred-LLM": .7345, "ImbLLM": .8002}

TABLE_III = {  # (sampling, permutation, fine-tuning) -> avg F1
    ("condition_y", "permute_xy", "major+minor"): .6833,
    ("condition_yx", "permute_xy", "major+minor"): .6690,
    ("condition_y", "fix_y", "major+minor"): .7408,
    ("condition_y", "permute_xy", "minor+interpolate"): .7436,
    ("condition_yx", "fix_y", "major+minor"): .7181,
    ("condition_yx", "permute_xy", "minor+interpolate"): .7998,
    ("condition_y", "fix_y", "minor+interpolate"): .7421,
    ("condition_yx", "fix_y", "minor+interpolate"): .8002,
}
TABLE_IV_AVG = {"ImbLLM-inter": {"F1": .7857, "AUC": .8359, "Close": .7871},
                "ImbLLM": {"F1": .8002, "AUC": .8503, "Close": .8011}}


def check_transcription() -> tuple[bool, dict]:
    res = {k: (round(float(np.mean(v)), 4), PRINTED_AVG[k]) for k, v in TABLE_II.items()}
    ok = all(abs(a - b) < 5e-5 + 1e-9 for a, b in res.values())
    return ok, res


def omitted_tests() -> dict:
    T = {k: np.array(v) for k, v in TABLE_II.items()}
    pw = {k: float(wilcoxon(T["ImbLLM"], T[k]).pvalue) for k in T if k != "ImbLLM"}
    adj = holm({k: v for k, v in pw.items() if k != "Original"})
    paired = {k: {"mean_diff": float((T["ImbLLM"] - T[k]).mean()),
                  "wins": int((T["ImbLLM"] > T[k]).sum()), "wilcoxon_p": pw[k],
                  "holm_p": adj.get(k)} for k in pw}
    comp = ("SMOTE-NC", "CTGAN", "TVAE", "GReaT", "TapTap", "Pred-LLM", "ImbLLM")
    M = np.array([T[c] for c in comp])
    chi2, p = friedmanchisquare(*M)
    ranks = np.array([rankdata(-M[:, j]) for j in range(M.shape[1])]).mean(0)
    cd = critical_difference(len(comp), len(DATASETS))
    best = ranks.min()
    return {"paired_vs_imbllm": paired, "friedman_chi2": float(chi2), "friedman_p": float(p),
            "mean_ranks": dict(zip(comp, ranks.round(3).tolist())), "cd": round(cd, 3),
            "not_separable_from_best": [c for c, r in zip(comp, ranks) if r - best < cd]}


def ablation_decomposition() -> dict:
    t = {k: v for k, v in TABLE_III.items()}
    base = t[("condition_y", "permute_xy", "major+minor")]
    full = t[("condition_yx", "fix_y", "minor+interpolate")]
    return {
        "C1_alone": round(t[("condition_yx", "permute_xy", "major+minor")] - base, 4),
        "C2_alone": round(t[("condition_y", "fix_y", "major+minor")] - base, 4),
        "C3_alone": round(t[("condition_y", "permute_xy", "minor+interpolate")] - base, 4),
        "C2_marginal_in_full": round(full - t[("condition_yx", "permute_xy",
                                               "minor+interpolate")], 4),
        "C1_marginal_in_full": round(full - t[("condition_y", "fix_y",
                                               "minor+interpolate")], 4),
        "C3_marginal_in_full": round(full - t[("condition_yx", "fix_y", "major+minor")], 4),
        "interpolation_marginal (Table IV)": round(TABLE_IV_AVG["ImbLLM"]["F1"]
                                                   - TABLE_IV_AVG["ImbLLM-inter"]["F1"], 4),
    }
