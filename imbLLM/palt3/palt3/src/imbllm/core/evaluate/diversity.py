"""Fidelity-diversity measurement and the conditioning entropy decomposition.

WHY IT IS BUILT THIS WAY (the palt2 version was degenerate)
  * palt2 took a plug-in entropy of the JOINT row code (8 bins ^ M features).
    With M >= ~8 almost every row is a unique pattern, so H ~= log(n): the
    "diversity" axis measured the sample count. Here every quantity is PER
    FEATURE (low-dimensional, estimable) and then averaged.
  * palt2 grouped rows by the exact prompt string. Under condition_yx nearly
    every prompt is unique, so every group had one row, within-group entropy
    was 0 and I(G;R) == H(mixture) on every run. Here R = (conditioned
    feature, value bin), which has a small number of levels with many rows each.
  * palt2's `identity_holds` checked (I - delta) == (H_yx - H_y), which is true
    by algebra for any numbers. It is gone; the decomposition is reported as a
    decomposition, with a Miller-Madow bias correction and a bootstrap CI.
  * fidelity is measured against HELD-OUT minority rows (palt2 used the
    generator's own training rows), and out-of-range synthetic mass lands in
    explicit overflow bins instead of being clipped into the edge bins.

THE DECOMPOSITION (replaces the paper's deleted Proposition 1)
    for each feature j:   H(G_j) = E_R[ H(G_j | R) ] + I(G_j ; R)
    Delta_H_j = H_yx(G_j) - H_y(G_j) = I(G_j ; R) - delta_j,
    delta_j   = H_y(G_j) - E_R[ H_yx(G_j | R) ]        (sharpening by conditioning)
    condition_yx raises diversity of feature j iff it moves modes (I) more
    than it sharpens them (delta). H_y comes from condition_y samples drawn from
    the SAME fine-tuned model (the model cache makes that a sampling-only cost).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.protocol import Schema


def _edges(ref: pd.Series, n_bins: int) -> np.ndarray:
    x = ref.astype(float).to_numpy()
    e = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(e) < 2:
        e = np.array([x.min() - 0.5, x.max() + 0.5])
    return np.concatenate([[-np.inf], e[1:-1], [np.inf]]) if len(e) > 2 else \
        np.array([-np.inf, e[0], np.inf])


def codes(df: pd.DataFrame, schema: Schema, ref: pd.DataFrame, n_bins: int = 8) -> dict:
    """feature -> integer codes, bins defined by the REFERENCE (real) data.
    Values outside the reference range fall in the first/last bin for continuous
    features and in an explicit 'unseen' code for categoricals."""
    out = {}
    for c in schema.continuous:
        e = _edges(ref[c], n_bins)
        out[c] = np.clip(np.searchsorted(e, df[c].astype(float).to_numpy(), side="right") - 1,
                         0, len(e) - 2)
    for c in schema.categorical:
        cats = sorted(ref[c].astype(str).unique())
        lut = {k: i for i, k in enumerate(cats)}
        out[c] = df[c].astype(str).map(lambda v: lut.get(v, len(cats))).to_numpy()
    return out


def entropy_mm(x: np.ndarray) -> float:
    """Plug-in entropy (nats) with the Miller-Madow correction (K-1)/(2n)."""
    if len(x) == 0:
        return float("nan")
    _, cnt = np.unique(x, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum() + (len(cnt) - 1) / (2 * len(x)))


def fidelity(synth: pd.DataFrame, real: pd.DataFrame, schema: Schema, ref: pd.DataFrame,
             n_bins: int = 8) -> dict:
    """Per-feature total variation between synthetic and REAL held-out rows."""
    cs, cr = codes(synth, schema, ref, n_bins), codes(real, schema, ref, n_bins)
    tv = {}
    for c in schema.features:
        k = int(max(cs[c].max(initial=0), cr[c].max(initial=0))) + 1
        ps = np.bincount(cs[c], minlength=k) / max(len(cs[c]), 1)
        pr = np.bincount(cr[c], minlength=k) / max(len(cr[c]), 1)
        tv[c] = float(0.5 * np.abs(ps - pr).sum())
    return {"tv_mean": float(np.mean(list(tv.values()))),
            "tv_max": float(max(tv.values())), "per_feature_tv": tv}


def diversity(synth: pd.DataFrame, schema: Schema, ref: pd.DataFrame, n_bins: int = 8) -> dict:
    """Mean per-feature entropy (nats) and normalised entropy in [0, 1]."""
    cs = codes(synth, schema, ref, n_bins)
    H = {c: entropy_mm(v) for c, v in cs.items()}
    norm = {c: H[c] / np.log(max(len(np.unique(v)), 2)) for c, v in cs.items()}
    return {"H_mean": float(np.mean(list(H.values()))),
            "H_norm_mean": float(np.mean(list(norm.values()))),
            "distinct_row_frac": float(synth[schema.features].astype(str)
                                       .drop_duplicates().shape[0] / max(len(synth), 1))}


def frontier_point(synth, heldout, schema, ref, n_bins: int = 8) -> dict:
    f = fidelity(synth, heldout, schema, ref, n_bins)
    d = diversity(synth, schema, ref, n_bins)
    return {"fidelity_tv_mean": f["tv_mean"], "diversity_H_mean": d["H_mean"],
            "diversity_H_norm": d["H_norm_mean"], "distinct_row_frac": d["distinct_row_frac"]}


def condition_groups(cond_col: list[str | None], cond_val: list, schema: Schema,
                     ref: pd.DataFrame, n_bins: int = 4) -> np.ndarray:
    """R for every generated row: '<feature>=<bin>' (None -> 'none')."""
    edges = {c: _edges(ref[c], n_bins) for c in schema.continuous}
    out = []
    for c, v in zip(cond_col, cond_val):
        if c is None:
            out.append("none")
        elif c in edges:
            b = int(np.clip(np.searchsorted(edges[c], float(v), side="right") - 1, 0,
                            len(edges[c]) - 2))
            out.append(f"{c}={b}")
        else:
            out.append(f"{c}={v}")
    return np.array(out)


def decompose(rows_yx: pd.DataFrame, groups: np.ndarray, rows_y: pd.DataFrame,
              schema: Schema, ref: pd.DataFrame, *, n_bins: int = 8, n_boot: int = 200,
              seed: int = 0, min_group: int = 5) -> dict:
    """Per-feature I(G_j;R), delta_j and Delta_H_j, averaged over features, with a
    bootstrap CI over rows. Features a group was conditioned ON are excluded for
    that group's contribution (their value was fixed by the prompt)."""
    cyx = codes(rows_yx, schema, ref, n_bins)
    cy = codes(rows_y, schema, ref, n_bins)
    groups = np.asarray(groups)

    def once(idx_yx, idx_y):
        I, D, DH = [], [], []
        g = groups[idx_yx]
        for c in schema.features:
            keep = np.array([not str(x).startswith(f"{c}=") for x in g])
            x = cyx[c][idx_yx][keep]
            gg = g[keep]
            levels, counts = np.unique(gg, return_counts=True)
            ok = counts >= min_group
            if ok.sum() < 2:
                continue
            mask = np.isin(gg, levels[ok])
            x, gg = x[mask], gg[mask]
            Hmix = entropy_mm(x)
            within = sum((gg == lv).mean() * entropy_mm(x[gg == lv]) for lv in levels[ok])
            Hy = entropy_mm(cy[c][idx_y])
            I.append(Hmix - within)
            D.append(Hy - within)
            DH.append(Hmix - Hy)
        return (np.mean(I), np.mean(D), np.mean(DH)) if I else (np.nan,) * 3

    base = once(np.arange(len(rows_yx)), np.arange(len(rows_y)))
    rng = np.random.default_rng(seed)
    boots = np.array([once(rng.integers(0, len(rows_yx), len(rows_yx)),
                           rng.integers(0, len(rows_y), len(rows_y))) for _ in range(n_boot)])
    ci = lambda j: [float(np.nanquantile(boots[:, j], .025)),  # noqa: E731
                    float(np.nanquantile(boots[:, j], .975))]
    levels = np.unique(groups, return_counts=True)
    return {"I_mean_nats": float(base[0]), "I_ci95": ci(0),
            "delta_mean_nats": float(base[1]), "delta_ci95": ci(1),
            "DeltaH_mean_nats": float(base[2]), "DeltaH_ci95": ci(2),
            "n_groups": int(len(levels[0])),
            "median_group_size": float(np.median(levels[1])),
            "verdict": "conditioning INCREASED per-feature diversity" if base[2] > 0
            else "conditioning did NOT increase per-feature diversity"}
