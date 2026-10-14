"""The ORACLE study: does utility tilting help when the generator is PERFECT?

No LLM. The minority class-conditional p_min is known in closed form, so the
"generator" draws exact samples from p_min. This isolates the sampling principle
of TILT from generator quality (docs/EXPERIMENTS.md, E0; docs/THEORY.md):

    Prop. 1 predicts   oracle oversampling (untilted) ≈ threshold_only
                       (a perfect generator only moves the threshold)
    Prop. 2 predicts   tilting by a variance utility (fisher / epistemic) gives a
                       FINITE-SAMPLE gain that shrinks as n_min grows, while an
                       aleatoric tilt (ambiguous rows) can hurt.

PROBLEM   d-dim. Majority ~ N(0, I). Minority ~ ½ N(+μ, I) + ½ N(-μ', I) (bimodal;
          `bimodal=False` -> a single mode). n_min real minority rows, IR · n_min majority.
ARMS      imbalance       learner on D, argmax
          threshold_only  learner on D, threshold tuned on a real validation set
          oracle          D + m exact p_min draws (m = n_maj - n_min), tuned threshold
          tilt_<u>@ε      D + m draws from q_λ ∝ p_min · e^{λ z_u}  (SIR from an exact
                          p_min pool of pool_factor·m, λ from the KL budget ε)
          smote           D + m SMOTE rows, tuned threshold
METRIC    minority F1 on a large test set; gain_vs_threshold = F1(arm) - F1(threshold_only)
LEARNERS  logistic regression (Prop. 2's setting) and XGBoost (misspecified for it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..core.data.protocol import Schema
from ..methods.tilt import tilt as T
from ..methods.tilt.utility import UtilityModel


@dataclass
class Problem:
    d: int = 4
    sep: float = 2.0
    bimodal: bool = True
    ir: float = 20.0

    def features(self) -> list[str]:
        return [f"x{i}" for i in range(self.d)]

    def sample_min(self, n: int, rng) -> np.ndarray:
        mu = np.zeros(self.d)
        mu[0] = self.sep
        if not self.bimodal:
            return rng.normal(size=(n, self.d)) + mu
        mu2 = np.zeros(self.d)
        mu2[1] = -self.sep
        pick = rng.random(n) < 0.5
        return rng.normal(size=(n, self.d)) + np.where(pick[:, None], mu, mu2)

    def sample_maj(self, n: int, rng) -> np.ndarray:
        return rng.normal(size=(n, self.d))

    def frame(self, n_min: int, rng, ir: float | None = None) -> pd.DataFrame:
        n_maj = int(round((ir or self.ir) * n_min))
        X = np.vstack([self.sample_min(n_min, rng), self.sample_maj(n_maj, rng)])
        df = pd.DataFrame(X, columns=self.features())
        df["y"] = ["yes"] * n_min + ["no"] * n_maj
        return df

    def schema(self) -> Schema:
        return Schema(target="y", continuous=self.features(), categorical=[],
                      classes=["no", "yes"], minority_label="yes", majority_label="no",
                      task="binary", source_task="binary")


def _learner(kind: str, seed: int):
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000)
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, verbosity=0,
                         random_state=seed, n_jobs=2)


def _tuned_f1(model, Xv, yv, Xt, yt) -> tuple[float, float]:
    pv = model.predict_proba(Xv)[:, 1]
    cand = np.unique(np.quantile(pv, np.linspace(0.5, 0.999, 200)))
    best = max(cand, key=lambda t: f1_score(yv, pv >= t, zero_division=0))
    pt = model.predict_proba(Xt)[:, 1]
    return float(f1_score(yt, pt >= best, zero_division=0)), float(best)


def _fit_eval(X, y, Xv, yv, Xt, yt, learner, seed, tune=True):
    m = _learner(learner, seed).fit(X, y)
    if not tune:
        return float(f1_score(yt, m.predict(Xt), zero_division=0))
    return _tuned_f1(m, Xv, yv, Xt, yt)[0]


def run_one(prob: Problem, n_min: int, seed: int, *, learners=("lr",),
            utilities=("fisher", "epistemic", "influence", "aleatoric"),
            budgets=(0.25,), pool_factor: float = 8.0, n_val: int = 2000,
            n_test: int = 20000) -> list[dict]:
    rng = np.random.default_rng(seed)
    feats = prob.features()
    D = prob.frame(n_min, rng)
    V = prob.frame(max(n_val // int(prob.ir + 1), 20), rng)      # real validation fold
    Te = prob.frame(max(n_test // int(prob.ir + 1), 200), rng)
    X, y = D[feats].to_numpy(), (D["y"] == "yes").to_numpy(int)
    Xv, yv = V[feats].to_numpy(), (V["y"] == "yes").to_numpy(int)
    Xt, yt = Te[feats].to_numpy(), (Te["y"] == "yes").to_numpy(int)
    m = int((y == 0).sum() - (y == 1).sum())
    pool = prob.sample_min(int(pool_factor * m), rng)
    pool_df = pd.DataFrame(pool, columns=feats)
    out = []
    for learner in learners:
        base = {"n_min": n_min, "seed": seed, "learner": learner, "ir": prob.ir,
                "bimodal": prob.bimodal}
        thr = _fit_eval(X, y, Xv, yv, Xt, yt, learner, seed)
        out.append({**base, "arm": "imbalance",
                    "f1": _fit_eval(X, y, Xv, yv, Xt, yt, learner, seed, tune=False)})
        out.append({**base, "arm": "threshold_only", "f1": thr})
        orc = prob.sample_min(m, rng)
        out.append({**base, "arm": "oracle", "f1": _fit_eval(
            np.vstack([X, orc]), np.r_[y, np.ones(m, int)], Xv, yv, Xt, yt, learner, seed)})
        try:
            from imblearn.over_sampling import SMOTE
            Xs, ys = SMOTE(k_neighbors=min(5, n_min - 1), random_state=seed).fit_resample(X, y)
            out.append({**base, "arm": "smote",
                        "f1": _fit_eval(Xs, ys, Xv, yv, Xt, yt, learner, seed)})
        except Exception:                                      # noqa: BLE001
            pass
        for u in utilities:
            um = UtilityModel(kind=u, seed=seed).fit_frame(D, prob.schema())
            z, _ = T.standardise(um.utility(pool_df))
            for eps in budgets:
                lam, cal = T.calibrate_lambda(z, eps, amp_cap=None)
                idx = T.select(lam * z, m, rng)
                Xa = np.vstack([X, pool[idx]])
                ya = np.r_[y, np.ones(len(idx), int)]
                out.append({**base, "arm": f"tilt_{u}@{eps:g}", "lambda": lam,
                            "kl": cal["kl"], "f1": _fit_eval(Xa, ya, Xv, yv, Xt, yt,
                                                              learner, seed)})
    for r in out:
        r["gain_vs_threshold"] = r["f1"] - next(
            o["f1"] for o in out if o["arm"] == "threshold_only" and o["learner"] == r["learner"])
    return out


def run(n_mins=(20, 50, 100), seeds=range(20), **kw) -> pd.DataFrame:
    prob_kw = {k: kw.pop(k) for k in ("d", "sep", "bimodal", "ir") if k in kw}
    prob = Problem(**prob_kw)
    rows = []
    for n in n_mins:
        for s in seeds:
            rows += run_one(prob, n, s, **kw)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["learner", "n_min", "arm"])["gain_vs_threshold"]
    out = g.agg(["mean", "std", "count"])
    out["ci95"] = 1.96 * out["std"] / np.sqrt(out["count"])
    return out.round(4)
