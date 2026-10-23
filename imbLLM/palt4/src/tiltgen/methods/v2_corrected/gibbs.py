"""FIX 4 -- order consistency: measure it, and (optionally) repair it.

DIAGNOSIS. Fine-tuning on random feature orders asks one network to represent
every conditional p(x_S | y, x_-S): a dependency network (Heckerman et al.,
JMLR 2000). Separately-learned conditionals need not be compatible with ANY
joint; with incompatible conditionals, ordered Gibbs sampling has a unique
stationary law (pseudo-Gibbs) that depends on the visiting order (their
Thm 3). A single left-to-right decode is one of M! factorisations.

MEASUREMENT -- order_sensitivity(). For a fixed row, score
log p(row | y) under several random feature orders (sum over value tokens +
EOS). A coherent joint gives the SAME number for every order; the spread (nats
per row / per token) is a direct measure of incompatibility.

REPAIR -- sir_gibbs(). For each feature j, in a fresh random order per sweep:
    context = "Y is y_min, <ALL other features in random order>, x_j is"
    so the causal LM's next-token distribution IS p(x_j | y, x_-j).
palt2 used only the features *preceding* j in the order, so a sweep simply
re-generated the row from scratch; that was not Gibbs sampling.
    categorical x_j  exact: every category is scored, x_j ~ p(. | y, x_-j)
    continuous  x_j  sampling-importance-resampling over a per-row candidate
                     set {current value} U {K draws from the minority KDE q}:
                     weight = p_model(v | y, x_-j) / q(v). Exact as K -> inf;
                     approximate for finite K (reported as such).
palt2 used one shared pool of 8 real values per feature for ALL rows, so every
refined continuous value was one of 8 copied training values.
    discrete  x_j  (flag snap_discrete, core/data/kinds.py) an integer column is
                     treated like a categorical: if its observed support S has at
                     most `max_exact` values, EVERY value of S is scored (exact
                     conditional, q uniform); otherwise the KDE draws are snapped
                     onto S and weighted by the KDE density at the snapped value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.data import kinds
from ...core.data.protocol import Splits
from ...core.serialize import LINK, SEP, clause, fmt_value, label_head


def _context(row: dict, others: list[str], col: str, schema, sig: int) -> str:
    parts = [label_head(schema)] + [clause(c, row[c], schema, sig) for c in others]
    return SEP.join(parts) + SEP + col + LINK.rstrip()


def sir_gibbs(rows: pd.DataFrame, engine, splits: Splits, rng: np.random.Generator, *,
              sweeps: int = 1, n_candidates: int = 16, max_rows: int | None = 256,
              sig: int = 5, snap_discrete: bool = False, max_exact: int = 64
              ) -> tuple[pd.DataFrame, dict]:
    from scipy.stats import gaussian_kde
    s = splits.schema
    if sweeps <= 0 or len(rows) == 0:
        return rows, {"sweeps": 0}
    out = rows.copy().reset_index(drop=True)
    sel = np.arange(len(out))
    if max_rows is not None and len(out) > max_rows:
        sel = np.sort(rng.choice(len(out), size=max_rows, replace=False))
    vocab = {c: sorted(splits.imbalanced_train[c].astype(str).unique()) for c in s.categorical}
    kdes = {}
    for c in s.continuous:
        v = splits.train_minor[c].astype(float).to_numpy()
        kdes[c] = gaussian_kde(v) if len(np.unique(v)) > 1 else None
    disc = set(s.discrete) if snap_discrete else set()
    sup = {c: kinds.support(splits.imbalanced_train, c) for c in disc}
    changed = {c: 0 for c in s.features}
    visits = {c: 0 for c in s.features}
    records = out.to_dict("records")

    for _ in range(sweeps):
        orders = {r: list(rng.permutation(s.features)) for r in sel}
        for pos in range(len(s.features)):
            ctxs, cands, logq, owners = [], [], [], []
            for r in sel:
                col = orders[r][pos]
                row = records[r]
                others = [c for c in orders[r] if c != col]
                if col in s.categorical:
                    vals = vocab[col]
                    lq = np.zeros(len(vals))
                elif col in disc and len(sup[col]) <= max_exact:
                    vals = [float(v) for v in sup[col]]          # exact over the support
                    lq = np.zeros(len(vals))
                else:
                    kde = kdes[col]
                    cur = float(row[col])
                    draws = kde.resample(n_candidates, seed=rng).ravel() if kde is not None \
                        else np.full(n_candidates, cur)
                    if col in disc:
                        cur = float(kinds.snap([cur], sup[col])[0])
                        draws = kinds.snap(draws, sup[col])
                    vals = [cur, *map(float, draws)]
                    lq = kde.logpdf(np.array(vals)) if kde is not None else np.zeros(len(vals))
                ctxs.append(_context(row, others, col, s, sig))
                cands.append([" " + fmt_value(v, col in s.continuous, sig) for v in vals])
                logq.append((col, vals, lq))
                owners.append(r)
            scores = engine.score(ctxs, cands, add_eos=True)
            for r, (col, vals, lq), sc in zip(owners, logq, scores):
                logw = np.asarray(sc, float) - lq
                if not np.isfinite(logw).any():
                    continue
                logw = np.where(np.isfinite(logw), logw, -np.inf)
                p = np.exp(logw - logw.max())
                pick = vals[int(rng.choice(len(vals), p=p / p.sum()))]
                visits[col] += 1
                if str(pick) != str(records[r][col]):
                    changed[col] += 1
                records[r][col] = pick
    out = pd.DataFrame(records)[s.features]
    return out, {"sweeps": sweeps, "rows_refined": int(len(sel)),
                 "n_candidates_continuous": n_candidates + 1,
                 "change_rate": {c: round(changed[c] / max(visits[c], 1), 4)
                                 for c in s.features},
                 "discrete_exact": sorted(c for c in disc if len(sup[c]) <= max_exact),
                 "note": "categorical: exact conditional; continuous: SIR with KDE proposal; "
                         "discrete: exact over the support, or snapped KDE draws"}


def _order_scores(rows: pd.DataFrame, engine, splits: Splits, rng: np.random.Generator,
                  n_rows: int, n_orders: int, sig: int) -> tuple[np.ndarray, np.ndarray]:
    """(n_rows, n_orders) summed log-probs and (n_rows,) token counts."""
    s = splits.schema
    take = rng.choice(len(rows), size=min(n_rows, len(rows)), replace=False)
    head = label_head(s) + ","
    ctxs, sufs = [], []
    for r in take:
        row = rows.iloc[int(r)].to_dict()
        texts = []
        for _ in range(n_orders):
            order = list(rng.permutation(s.features))
            texts.append(" " + SEP.join(clause(c, row[c], s, sig) for c in order))
        ctxs.append(head)
        sufs.append(texts)
    scores = np.array(engine.score(ctxs, sufs, add_eos=True), float)
    # token count of the same clauses in a different order differs by at most a few
    # tokens under byte-level BPE; use each row's mean over its orders
    n_tok = np.array([np.mean([len(x) for x in engine.encode(t, eos=True)]) for t in sufs])
    return scores, n_tok


def _summary(scores: np.ndarray, n_tok: np.ndarray, rng: np.random.Generator,
             n_boot: int = 1000) -> dict:
    per_tok_std = scores.std(1, ddof=1) / n_tok          # primary statistic
    boot = [per_tok_std[rng.integers(0, len(per_tok_std), len(per_tok_std))].mean()
            for _ in range(n_boot)]
    return {"n_rows": int(len(scores)),
            "std_nats_per_token_mean": float(per_tok_std.mean()),
            "std_nats_per_token_ci95": [float(np.percentile(boot, 2.5)),
                                        float(np.percentile(boot, 97.5))],
            "std_nats_per_row_mean": float(scores.std(1, ddof=1).mean()),
            "range_nats_per_row_mean": float((scores.max(1) - scores.min(1)).mean()),
            "logp_per_token_mean": float((scores.mean(1) / n_tok).mean())}


def order_sensitivity(rows: pd.DataFrame, engine, splits: Splits, rng: np.random.Generator, *,
                      n_rows: int = 128, n_orders: int = 12, sig: int = 5,
                      reference: pd.DataFrame | None = None) -> dict:
    """Spread of log p(row | y) across random feature orders, WITH a null reference.

    Any imperfectly fitted model has a non-zero spread, so the synthetic-row number
    means nothing on its own (docs/RESEARCH_REVIEW.md 3.3). It is therefore
    reported next to the SAME statistic on REAL held-out minority rows scored by the
    same model (`reference`, default D*_minor minus D_minor), and as a ratio.
    Comparisons that make it a finding (experiment `order`):
        * permuted-order fine-tune vs a fixed-order fine-tune (`permutation=identity`),
        * larger backbone / longer training -> does the spread shrink?
        * per-row spread vs discriminator / downstream effect (`tilt report order`).
    Primary statistic: per-row std across orders divided by token count (nats per
    token), averaged over rows, with a bootstrap 95% CI over rows. The max-min range
    is also reported but grows with n_orders, so it is not comparable across runs
    with different n_orders."""
    if len(rows) == 0:
        return {"error": "no rows"}
    if n_orders < 2:
        raise ValueError("n_orders must be >= 2")
    sc, nt = _order_scores(rows, engine, splits, rng, n_rows, n_orders, sig)
    out = {"n_orders": n_orders, "synthetic": _summary(sc, nt, rng)}
    ref = reference if reference is not None else splits.minor_heldout
    ref_name = "real_heldout_minority" if reference is None else "reference"
    if ref is None or len(ref) == 0:
        ref, ref_name = splits.train_minor, "real_train_minority (heldout empty)"
    rs, rn = _order_scores(ref[splits.schema.features], engine, splits, rng, n_rows,
                           n_orders, sig)
    out["real"] = _summary(rs, rn, rng)
    out["real_source"] = ref_name
    denom = out["real"]["std_nats_per_token_mean"]
    out["ratio_synthetic_to_real"] = (out["synthetic"]["std_nats_per_token_mean"] / denom
                                      if denom > 0 else float("nan"))
    out["interpretation"] = ("0 => every decoding order assigns the row the same probability "
                             "(a coherent joint). Compare `synthetic` with `real` (same model) "
                             "and with a fixed-order fine-tune (experiment `order`); a spread "
                             "is only evidence of order-dependence relative to those.")
    return out
