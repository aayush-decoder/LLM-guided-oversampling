"""u(x): how useful a synthetic minority row x would be to the downstream learner.

TILT samples from  q*(x) ∝ p_min(x) · exp(λ u(x)) · 1[x is label-valid]  (docs/METHOD.md).
This module supplies u and the validity test. Every model here is fitted on the
REAL imbalanced training set D_train only (never on val / test / held-out rows),
with 2-fold CROSS-FITTING: the probe that scores a quantity is never the probe
fitted on the fold that quantity is measured on.

PROBE  standardised features (continuous z-scored, categoricals one-hot), plus an
       intercept, x~ = [x_std, 1];  L2 logistic regression  σ(θᵀx~) = P(minority | x).

UTILITIES (docs/THEORY.md, Proposition 2 gives the first two)
    fisher      u2(x) = σ(1-σ) · x~ᵀ H⁻¹ H_val H⁻¹ x~       (DEFAULT)
                first-order reduction of the held-out variance proxy
                tr(H_val (H + ε I(x))⁻¹) from adding x; LABEL-FREE; peaks at the
                boundary AND where the data are sparse (large H⁻¹ leverage).
    influence   u1(x) = (1-σ(x)) · gᵀ H⁻¹ x~,   g = -∇R_val(θ̂)
                first-order decrease of the class-balanced validation log-loss
                when x is added WITH LABEL minority (can be negative = harmful).
    epistemic   std over B bootstrap probes of logit P(minority | x)
                (where more data would change the model; model-agnostic)
    aleatoric   4 p(1-p) with p from the class-balanced probe
                (ABLATION: expected to HURT; ambiguous rows add label noise,
                 the objection CLLM-style curation is built on)
    none        u ≡ 0  -> TILT reduces to the untilted generator (v2 recommended)

VALIDITY (answers the label-noise objection)
    A row is label-valid if a class-balanced NON-LINEAR probe (histogram gradient
    boosting, class-balanced sample weights) gives it
        p_bal(x) >= τ_valid,  τ_valid = the `valid_q` quantile (default 5%) of
    p_bal on REAL minority rows, out of fold. Tilting toward uncertainty can
    therefore never select a row that looks less minority-like than 95% of
    real minority rows. The probe is non-linear on purpose: a linear probe cannot
    separate a multi-modal minority from a majority that sits between its modes
    (tests/test_tilt.py found 57% of majority rows "valid" with a linear probe).
    The `aleatoric` utility uses the same probe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...core.data.protocol import Schema, Splits

UTILITIES = ("fisher", "influence", "epistemic", "aleatoric", "none")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class Design:
    """Row frame -> standardised design matrix with intercept (fitted on D_train)."""

    def __init__(self, schema: Schema):
        self.s = schema

    def fit(self, df: pd.DataFrame) -> "Design":
        s = self.s
        X = df[s.continuous].astype(float).to_numpy() if s.continuous else np.zeros((len(df), 0))
        self.mu = X.mean(0) if len(X) else np.zeros(0)
        sd = X.std(0) if len(X) else np.zeros(0)
        self.sd = np.where(sd > 0, sd, 1.0)
        self.cats = {c: sorted(df[c].astype(str).unique()) for c in s.categorical}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        s = self.s
        blocks = []
        if s.continuous:
            X = df[s.continuous].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            blocks.append(np.nan_to_num((X - self.mu) / self.sd))
        for c in s.categorical:
            v = df[c].astype(str).to_numpy()
            blocks.append((v[:, None] == np.array(self.cats[c])[None, :]).astype(float))
        blocks.append(np.ones((len(df), 1)))
        return np.hstack(blocks)


def _fit_lr(X: np.ndarray, y: np.ndarray, *, l2: float, weights: np.ndarray | None = None,
            iters: int = 50) -> np.ndarray:
    """L2 logistic regression by Newton's method (intercept = last column, not
    penalised). Returns θ. Plain numpy so H is exactly the Hessian we use."""
    n, d = X.shape
    w = np.ones(n) if weights is None else weights
    w = w * n / w.sum()
    R = l2 * np.eye(d)
    R[-1, -1] = 0.0
    th = np.zeros(d)
    for _ in range(iters):
        p = _sigmoid(X @ th)
        grad = X.T @ (w * (p - y)) / n + R @ th
        H = (X * (w * p * (1 - p))[:, None]).T @ X / n + R + 1e-9 * np.eye(d)
        step = np.linalg.solve(H, grad)
        th -= step
        if np.abs(step).max() < 1e-8:
            break
    return th


def _hessian(X: np.ndarray, th: np.ndarray, l2: float, weights=None) -> np.ndarray:
    n, d = X.shape
    w = np.ones(n) if weights is None else weights * n / weights.sum()
    p = _sigmoid(X @ th)
    R = l2 * np.eye(d)
    R[-1, -1] = 0.0
    return (X * (w * p * (1 - p))[:, None]).T @ X / n + R + 1e-9 * np.eye(d)


def _fit_hgb(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """Class-balanced histogram gradient boosting (the validity / aleatoric probe)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    if len(np.unique(y)) < 2:
        return None
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed
                                          ).fit(X, y, sample_weight=_balanced_weights(y))


def _p_hgb(model, X: np.ndarray) -> np.ndarray:
    return np.full(len(X), 0.5) if model is None else model.predict_proba(X)[:, 1]


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    pos = max(y.mean(), 1e-9)
    return np.where(y == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-9))


@dataclass
class UtilityModel:
    kind: str = "fisher"
    l2: float = 1e-2
    n_boot: int = 8
    valid_q: float = 0.05
    seed: int = 0
    folds: list = field(default_factory=list, repr=False)

    def fit(self, splits: Splits) -> "UtilityModel":
        return self.fit_frame(splits.imbalanced_train, splits.schema)

    def fit_frame(self, D: pd.DataFrame, s: Schema) -> "UtilityModel":
        """Fit on a labelled frame D (features + s.target). Used directly by the
        no-LLM oracle study (experiments/oracle.py)."""
        if self.kind not in UTILITIES:
            raise ValueError(f"utility must be one of {UTILITIES}")
        D = D.reset_index(drop=True)
        self.design = Design(s).fit(D)
        X = self.design.transform(D)
        y = (D[s.target].astype(str) == s.minority_label).to_numpy(float)
        rng = np.random.default_rng(self.seed)
        # stratified 2-fold split
        idx = np.arange(len(y))
        fold = np.zeros(len(y), int)
        for cls in (0, 1):
            ii = rng.permutation(idx[y == cls])
            fold[ii[len(ii) // 2:]] = 1
        self.folds = []
        oof_pbal = np.full(len(y), np.nan)
        for a in (0, 1):
            A, B = fold == a, fold != a
            th = _fit_lr(X[A], y[A], l2=self.l2)
            H = _hessian(X[A], th, self.l2)
            Hval = _hessian(X[B], th, self.l2)
            wb = _balanced_weights(y[B])
            pB = _sigmoid(X[B] @ th)
            g = -(X[B].T @ (wb * (pB - y[B]))) / wb.sum()        # -∇R_val (balanced log-loss)
            th_bal = _fit_lr(X[A], y[A], l2=self.l2, weights=_balanced_weights(y[A]))
            hgb = _fit_hgb(X[A], y[A], seed=self.seed)
            boots = []
            if self.kind == "epistemic":
                XA, yA = X[A], y[A]
                for _ in range(self.n_boot):
                    bi = rng.integers(0, len(yA), len(yA))
                    boots.append(_fit_lr(XA[bi], yA[bi], l2=self.l2, iters=25))
            Hinv = np.linalg.inv(H)
            self.folds.append({"th": th, "Hinv": Hinv, "M": Hinv @ Hval @ Hinv,
                               "g": g, "th_bal": th_bal, "hgb": hgb, "boots": boots})
            oof_pbal[B] = _p_hgb(hgb, X[B])
        m = y == 1
        self.tau_valid = float(np.quantile(oof_pbal[m], self.valid_q)) if m.any() else 0.0
        return self

    # ------------------------------------------------------------------ scoring
    def _per_fold(self, X: np.ndarray, f: dict) -> np.ndarray:
        if self.kind == "none":
            return np.zeros(len(X))
        p = _sigmoid(X @ f["th"])
        if self.kind == "fisher":
            return p * (1 - p) * np.einsum("ij,jk,ik->i", X, f["M"], X)
        if self.kind == "influence":
            return (1 - p) * (X @ (f["Hinv"] @ f["g"]))
        if self.kind == "aleatoric":
            pb = _p_hgb(f["hgb"], X)
            return 4 * pb * (1 - pb)
        if self.kind == "epistemic":
            L = np.stack([X @ b for b in f["boots"]], 1)
            return L.std(1)
        raise ValueError(self.kind)

    def utility(self, rows: pd.DataFrame) -> np.ndarray:
        X = self.design.transform(rows)
        return np.mean([self._per_fold(X, f) for f in self.folds], 0)

    def p_probe(self, rows: pd.DataFrame) -> np.ndarray:
        """P(minority | x) of the linear probe that defines the fisher / influence utilities."""
        X = self.design.transform(rows)
        return np.mean([_sigmoid(X @ f["th"]) for f in self.folds], 0)

    def p_balanced(self, rows: pd.DataFrame) -> np.ndarray:
        X = self.design.transform(rows)
        return np.mean([_p_hgb(f["hgb"], X) for f in self.folds], 0)

    def valid(self, rows: pd.DataFrame) -> np.ndarray:
        return self.p_balanced(rows) >= self.tau_valid
