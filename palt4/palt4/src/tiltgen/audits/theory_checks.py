"""Numerical checks of the thesis propositions (docs/THEORY.md). No LLM, no data.

Each check returns (name, passed, details). They verify SHIPPED code where the
proposition is about code (tilt calibration, twisted SMC) and a simulation where
it is about learning (population equivalence, first-order gain).

    lemma1_gibbs          q* = p e^{λu}/Z beats random q on E_q[u] - KL(q||p)/λ
    prop1_population      a PERFECT generator (+ tuned threshold) ≈ threshold alone;
                          untuned, it shifts the LR intercept by log((n_min+m)/n_min)
    prop2_first_order     predicted ΔR = -ε E_q[u1] vs the actual refit ΔR on the
                          balanced validation log-loss, across several tilts q
    prop3_smc_exact       field-wise twisted SMC (methods/tilt/smc.py) reproduces
                          q* ∝ p e^{λu} on a 2-field toy row model, with an
                          UNINFORMATIVE twist and with an informative one
    prop4_amplification   calibrate_lambda never exceeds the amplification cap
    budget_monotone       calibrated λ gives KL <= ε, and KL grows with ε
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.data.protocol import Schema
from ..methods.tilt import smc as S
from ..methods.tilt import tilt as T


def lemma1_gibbs(seed: int = 0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(12))
    u = rng.normal(size=12)
    lam = 1.7
    q = p * np.exp(lam * u)
    q /= q.sum()
    obj = lambda r: float(r @ u - np.sum(r * np.log(r / p)) / lam)  # noqa: E731
    best = obj(q)
    worst_gap = min(best - obj(rng.dirichlet(np.ones(12))) for _ in range(2000))
    return "lemma1_gibbs", worst_gap >= -1e-9, {"objective_at_q*": best,
                                               "min_gap_vs_2000_random_q": worst_gap}


def prop1_population(seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    d, n_min, n_maj = 3, 3000, 60000
    mu = np.r_[1.5, np.zeros(d - 1)]
    X = np.vstack([rng.normal(size=(n_min, d)) + mu, rng.normal(size=(n_maj, d))])
    y = np.r_[np.ones(n_min), np.zeros(n_maj)]
    m = n_maj - n_min
    Xa = np.vstack([X, rng.normal(size=(m, d)) + mu])
    ya = np.r_[y, np.ones(m)]
    Xt = np.vstack([rng.normal(size=(20000, d)) + mu, rng.normal(size=(400000, d))])
    yt = np.r_[np.ones(20000), np.zeros(400000)]
    base = LogisticRegression(C=1e6, max_iter=3000).fit(X, y)
    over = LogisticRegression(C=1e6, max_iter=3000).fit(Xa, ya)

    def best_f1(model):
        p = model.predict_proba(Xt)[:, 1]
        return max(f1_score(yt, p >= t) for t in np.quantile(p, np.linspace(0.9, 0.999, 120)))

    f_base, f_over = best_f1(base), best_f1(over)
    shift = float(over.intercept_[0] - base.intercept_[0])
    theory = float(np.log((n_min + m) / n_min))
    ok = abs(f_base - f_over) < 0.01 and abs(shift - theory) / theory < 0.1
    return "prop1_population", ok, {"f1_threshold_only": f_base, "f1_perfect_oversampling":
                                    f_over, "intercept_shift": shift,
                                    "theory_log_ratio": theory}


def _lr(X, y, w=None, l2=1e-3):
    from ..methods.tilt.utility import _fit_lr
    return _fit_lr(X, y, l2=l2, weights=w)


def prop2_first_order(seed: int = 0):
    from ..methods.tilt.utility import _balanced_weights, _hessian, _sigmoid
    rng = np.random.default_rng(seed)
    d, n_min, n_maj, l2 = 3, 60, 1200, 1e-3
    mu = np.r_[1.2, -0.8, 0.0]
    make = lambda a, b: (np.vstack([rng.normal(size=(a, d)) + mu, rng.normal(size=(b, d))]),  # noqa: E731
                         np.r_[np.ones(a), np.zeros(b)])
    X, y = make(n_min, n_maj)
    Xv, yv = make(4000, 80000)
    add1 = lambda A: np.hstack([A, np.ones((len(A), 1))])  # noqa: E731
    X1, Xv1 = add1(X), add1(Xv)
    th = _lr(X1, y, l2=l2)
    H = _hessian(X1, th, l2)
    wv = _balanced_weights(yv)
    R = lambda t: float(np.sum(wv * -(yv * np.log(_sigmoid(Xv1 @ t) + 1e-12) + (1 - yv)  # noqa: E731
                                      * np.log(1 - _sigmoid(Xv1 @ t) + 1e-12))) / wv.sum())
    pv = _sigmoid(Xv1 @ th)
    g = -(Xv1.T @ (wv * (pv - yv))) / wv.sum()
    pool = rng.normal(size=(20000, d)) + mu
    P1 = add1(pool)
    u1 = (1 - _sigmoid(P1 @ th)) * (P1 @ np.linalg.solve(H, g))
    z, _ = T.standardise(u1)
    n = len(y)
    out = {}
    for eps_w in (0.002, 0.05):                 # synthetic mass: first-order vs balancing scale
        preds, acts = [], []
        for lam in (-2.0, -1.0, 0.0, 1.0, 2.0):
            w = T.normalised(lam * z)
            idx = rng.choice(len(pool), size=2000, p=w)
            Xa = np.vstack([X1, P1[idx]])
            ya = np.r_[y, np.ones(len(idx))]
            wa = np.r_[np.ones(n), np.full(len(idx), eps_w * n / len(idx))]
            # objective (1/n)[sum_real + eps n mean_syn] -> _fit_lr normalises total weight
            # to n(1+eps), so its ridge term is rescaled to keep the same penalty
            th_a = _lr(Xa, ya, w=wa, l2=l2 / (1 + eps_w))
            acts.append(R(th_a) - R(th))
            preds.append(-eps_w * float(np.mean(u1[idx])))
        out[f"eps={eps_w:g}"] = {"predicted_dR": preds, "actual_dR": acts,
                                 "corr": float(np.corrcoef(preds, acts)[0, 1]),
                                 "slope": float(np.polyfit(preds, acts, 1)[0])}
    small = out["eps=0.002"]
    ok = small["corr"] > 0.95 and 0.7 <= small["slope"] <= 1.3
    out["reading"] = ("first-order law holds for small synthetic mass; at eps=0.05 the "
                      "response saturates (slope << 1): the reason for a KL budget")
    return "prop2_first_order", ok, out


class _ToyRowLM:
    """Fields a then b, values '0'/'1': p(a=1)=0.3, p(b=1|a)=0.8 if a=1 else 0.2."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def complete(self, prompts, **kw):
        out = []
        for p in prompts:
            if "a is" not in p:
                a = int(self.rng.random() < 0.3)
                out.append((f" a is {a}, b is ", False))
            else:
                a = int(p.split("a is ")[1][0])
                b = int(self.rng.random() < (0.8 if a else 0.2))
                out.append((f" b is {b}", True))
        return out


def _toy_schema():
    return Schema(target="y", continuous=[], categorical=["a", "b"], classes=["no", "yes"],
                  minority_label="yes", majority_label="no", task="binary",
                  source_task="binary")


def prop3_smc_exact(seed: int = 0, groups: int = 5000):
    s = _toy_schema()
    vocab = {"a": ["0", "1"], "b": ["0", "1"]}
    lam = 1.2
    u = lambda a, b: float(a) - float(b)  # noqa: E731
    p = {(a, b): (0.3 if a else 0.7) * ((0.8 if b else 0.2) if a else (0.2 if b else 0.8))
         for a in (0, 1) for b in (0, 1)}
    q = {k: v * np.exp(lam * u(*k)) for k, v in p.items()}
    Z = sum(q.values())
    q = {k: v / Z for k, v in q.items()}
    term = lambda df: lam * (df["a"].astype(int) - df["b"].astype(int)).to_numpy(float)  # noqa: E731
    res = {}
    for name, twist in (("flat_twist", lambda rows: np.zeros(len(rows))),
                        ("informed_twist", lambda rows: np.array(
                            [lam * (float(r["a"]) - 0.5) for r in rows]))):
        # the toy LM and the sampler must NOT share a seed: identical streams make the
        # Gumbel keys correlated with the LM's draws and bias the check, not the sampler
        frame, st = S.twisted_smc(_ToyRowLM(seed + 1000), ["y is yes,"] * groups, s,
                                  vocab=vocab, repair=False, twist=twist, terminal=term,
                                  n_particles=16, keep=1, dedupe=False,
                                  rng=np.random.default_rng(seed), seed=seed)
        emp = frame.groupby(["a", "b"]).size() / len(frame)
        tv = 0.5 * sum(abs(emp.get((str(a), str(b)), 0.0) - q[(a, b)])
                       for a in (0, 1) for b in (0, 1))
        res[name] = {"tv_to_q*": float(tv), "rows": int(len(frame)),
                     "resamples": st.resamples}
    ok = all(r["tv_to_q*"] < 0.04 for r in res.values())      # ~2x the sampling noise
    return "prop3_smc_exact", ok, res


def prop4_amplification(seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=5000)
    z = np.clip(z, -3, 3)
    lam, info = T.calibrate_lambda(z, epsilon=10.0, amp_cap=5.0)
    amp = float(np.exp(lam * (z.max() - z.min())))
    return "prop4_amplification", amp <= 5.0 + 1e-6 and info["binding"] == "amplification", \
        {"lambda": lam, "max_amplification": amp, "binding": info["binding"]}


def budget_monotone(seed: int = 0):
    rng = np.random.default_rng(seed)
    z, _ = T.standardise(rng.gamma(2.0, size=4000))
    kls = []
    for eps in (0.05, 0.1, 0.25, 0.5, 1.0):
        lam, info = T.calibrate_lambda(z, eps)
        kls.append(info["kl"])
    ok = all(k <= e + 1e-3 for k, e in zip(kls, (0.05, 0.1, 0.25, 0.5, 1.0))) and \
        all(a <= b + 1e-9 for a, b in zip(kls, kls[1:]))
    return "budget_monotone", ok, {"kl_at_budgets": kls}


CHECKS = (lemma1_gibbs, prop1_population, prop2_first_order, prop3_smc_exact,
          prop4_amplification, budget_monotone)


def run_all(seed: int = 0):
    return [c(seed) for c in CHECKS]


__all__ = ["run_all", "CHECKS", "pd"]
