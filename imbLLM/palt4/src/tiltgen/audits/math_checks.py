"""Math audits that exercise the SHIPPED code (palt2's audits re-implemented the
maths inline, so a passing audit said nothing about the code that actually ran).

Each check returns (name, passed, details). No LLM, no downloads, ~seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.data.protocol import Schema, Splits
from ..core.evaluate import quality
from ..core.evaluate import diversity
from ..methods.v1_paper import interpolate as v1i
from ..methods.v1_paper import permute
from ..methods.v2_corrected import interpolate as v2i
from ..methods.v2_corrected.condition import TauBridge


def _schema(d: int, cats: list[str] | None = None) -> Schema:
    return Schema(target="y", continuous=[f"x{i}" for i in range(d)], categorical=cats or [],
                  classes=["a", "b"], minority_label="b", majority_label="a",
                  task="binary", source_task="binary")


def _gauss(n, d, rng):
    A = rng.normal(size=(d, d))
    L = np.linalg.cholesky(A @ A.T / d + 1e-6 * np.eye(d))
    return pd.DataFrame(rng.normal(size=(n, d)) @ L.T, columns=[f"x{i}" for i in range(d)])


def _bimodal(n, rng):
    c = rng.integers(0, 2, n)
    P = np.where(c[:, None] == 0, rng.normal([-3, -3], .4, (n, 2)), rng.normal([3, 3], .4, (n, 2)))
    return pd.DataFrame(P, columns=["x0", "x1"])


def covariance_contraction(rng=None):
    """v1 Eq.(5) as shipped: Cov ratio == 2/3 - 1/(3(n-1)) within 0.02."""
    rng = rng or np.random.default_rng(7)
    rows, ok = [], True
    for n in (50, 200, 2000):
        for d in (2, 5, 10):
            X = _gauss(n, d, rng)
            r = np.mean([v1i.covariance_ratio(X, v1i.interpolate(X, _schema(d), n, rng)[0],
                                              list(X.columns)) for _ in range(60)])
            th = v1i.theoretical_covariance_ratio(n)
            ok &= abs(r - th) < 0.02
            rows.append({"n": n, "d": d, "empirical": round(float(r), 4), "theory": round(th, 4)})
    return "covariance: random-partner Cov -> (2/3)Sigma", bool(ok), rows


def geometry_table(rng=None):
    """The FIX-B1 table, produced by the shipped v2 interpolate(); reproducible."""
    rng = rng or np.random.default_rng(11)
    out = {}
    for regime, X in (("unimodal", _gauss(600, 2, rng)), ("bimodal", _bimodal(600, rng))):
        sch = _schema(2)
        for m in v2i.MODES:
            Z, _, _ = v2i.interpolate(X, sch, 6000, rng, mode=m)
            out.setdefault(m, {})[regime] = {
                "cov_ratio": round(v1i.covariance_ratio(X, Z, sch.continuous), 3),
                "off_manifold": round(v2i.off_manifold_fraction(X, Z, sch.continuous), 4)}
    ok = (out["random"]["bimodal"]["off_manifold"] > 0.2
          and out["knn"]["bimodal"]["off_manifold"] < 0.02
          and abs(out["random"]["unimodal"]["cov_ratio"] - 2 / 3) < 0.05
          and abs(out["variance_preserving"]["unimodal"]["cov_ratio"] - 1) < 0.1)
    return "geometry: spread vs manifold for the 5 interpolation modes", bool(ok), out


def tau_endpoints(rng=None):
    """tau=0 draws match the MINORITY empirical law, tau=1 the POOLED one (measured)."""
    rng = rng or np.random.default_rng(3)
    maj = pd.DataFrame({"x0": rng.normal(0, 1, 900), "c0": rng.choice(list("pqr"), 900,
                                                                    p=[.7, .2, .1])})
    mi = pd.DataFrame({"x0": rng.normal(2.5, .5, 60), "c0": rng.choice(list("pqr"), 60,
                                                                     p=[.1, .3, .6])})
    pooled = pd.concat([maj, mi], ignore_index=True)
    sch = _schema(1, ["c0"])
    kl0 = TauBridge(sch, mi, pooled, 0.0).measured_kl()
    kl1 = TauBridge(sch, mi, pooled, 1.0).measured_kl()
    kls = [TauBridge(sch, mi, pooled, t).measured_kl()["mean_kl_to_minority"]
           for t in (0, .25, .5, .75, 1)]
    # tau=0 is compared against a 60-row minority histogram, so ~0.03 nats is
    # finite-sample noise; the monotone rise to ~1.3 nats at tau=1 is the signal.
    ok = kl0["mean_kl_to_minority"] < 0.06 and kl1["mean_kl_to_pooled"] < 0.02 \
        and all(b >= a - 1e-3 for a, b in zip(kls, kls[1:])) and kls[-1] > kls[0] + 0.3
    return "tau-bridge: endpoints exact, measured from draws", bool(ok), \
        {"tau0": kl0["mean_kl_to_minority"], "tau1_to_pooled": kl1["mean_kl_to_pooled"],
         "kl_to_minority_by_tau": [round(x, 4) for x in kls]}


def entropy_identity():
    """H(mix) = E_R H(G|R) + I(G;R): the sign of Delta_H = I - delta is not fixed."""
    def H(p):
        p = np.asarray(p, float)
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    V, nR = 8, 6
    pr = np.full(nR, 1 / nR)
    base = np.zeros(V)
    base[[2, 3, 0]] = [.9, .07, .03]
    moving = np.array([np.eye(V)[j % V] * .93 + .01 for j in range(nR)])
    moving /= moving.sum(1, keepdims=True)
    sharp = np.array([base ** 1.5 / (base ** 1.5).sum()] * nR)
    res = {}
    for tag, qs in (("mode_moving", moving), ("sharpening", sharp)):
        mix = (pr[:, None] * qs).sum(0)
        within = float((pr * np.array([H(q) for q in qs])).sum())
        res[tag] = {"I": round(H(mix) - within, 4), "delta": round(H(base) - within, 4),
                    "DeltaH": round(H(mix) - H(base), 4)}
    ok = res["mode_moving"]["DeltaH"] > 0 > res["sharpening"]["DeltaH"]
    return "entropy: Delta_H = I(G;R) - delta can take either sign", bool(ok), res


def decomposition_estimator(rng=None):
    """The shipped per-feature estimator recovers a KNOWN I(G;R) (the palt2 one
    returned I = H(mixture) whenever prompts were unique)."""
    rng = rng or np.random.default_rng(5)
    n = 4000
    R = rng.integers(0, 4, n)
    x = rng.normal(R * 1.5, 1.0)                     # depends on R
    z = rng.normal(0, 1, n)                           # independent of R
    rows = pd.DataFrame({"x0": x, "x1": z})
    ref = pd.DataFrame({"x0": rng.normal(2.25, 2, 3000), "x1": rng.normal(0, 1, 3000)})
    rows_y = pd.DataFrame({"x0": rng.normal(2.25, 2.0, n), "x1": rng.normal(0, 1, n)})
    groups = np.array([f"c={r}" for r in R])
    sch = _schema(2)
    d = diversity.decompose(rows, groups, rows_y, sch, ref, n_boot=50)
    per = {}
    for c in ("x0", "x1"):
        one = _schema(1)
        one.continuous = [c]
        per[c] = diversity.decompose(rows[[c]], groups, rows_y[[c]], one, ref[[c]],
                                     n_boot=20)["I_mean_nats"]
    ok = per["x0"] > 0.3 and abs(per["x1"]) < 0.05
    return "decomposition: estimator finds I>0 only where G depends on R", bool(ok), \
        {"I_dependent_feature": round(per["x0"], 4), "I_independent_feature": round(per["x1"], 4),
         "overall": {k: v for k, v in d.items() if k != "verdict"}}


def label_free():
    vals = {m: permute.label_free_fraction(m, "permute_xy") for m in (4, 11, 20, 40)}
    ok = all(abs(v - .5) < 1e-12 for v in vals.values())
    return "permutation: exactly 1/2 of features trained label-free under permute_xy", ok, vals


def close_probability_counterexample(rng=None):
    """B uniformly closer than A, yet palt2's per-method max-normalisation ranks
    A higher. The fixed normaliser ranks B higher."""
    rng = rng or np.random.default_rng(0)
    real = rng.random((200, 3))
    A = real + rng.normal(0, .004, real.shape)
    A[0] = real[0] + 5.0                              # A misses ONE real row badly
    B = real + rng.normal(0, .003, real.shape)        # B is closer on EVERY row
    dA, dB = quality.nn_dist(A, real), quality.nn_dist(B, real)
    palt2 = lambda d: float((d / d.max() <= .2).mean())  # noqa: E731
    fixed_a, fixed_b = float((dA <= .01).mean()), float((dB <= .01).mean())
    ok = palt2(dA) > palt2(dB) and fixed_b >= fixed_a
    return "close probability: per-method max-normalisation inverts rankings", bool(ok), \
        {"palt2_A": palt2(dA), "palt2_B": palt2(dB), "fixed_A": fixed_a, "fixed_B": fixed_b}


ALL = (covariance_contraction, geometry_table, tau_endpoints, entropy_identity,
       decomposition_estimator, label_free, close_probability_counterexample)


def run_all() -> list[tuple[str, bool, object]]:
    return [f() for f in ALL]


__all__ = ["ALL", "run_all", "Splits"]
