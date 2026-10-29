"""Figures. Matplotlib (Agg), written to disk, never shown -- works headless.

    data_report(name)     BEFORE / AFTER cleaning and the protocol, per dataset:
        01_funnel.png         raw -> cleaned -> train/val/test -> D*_minor -> D_minor
        02_class_balance.png  class counts before vs after the q-reduction
        03_features_raw_vs_clean.png   per-feature distributions, raw file vs cleaned
        04_minority_geometry.png       PCA of the continuous block: majority,
                              D_minor, held-out minority, and the interpolated
                              fine-tuning rows under v1 (random) vs v2 (knn)
    synth_report(run_dir) a finished run: real held-out vs synthetic marginals,
                          PCA overlay, and the DCR histogram (paper Fig. 7)
    report_figure(exp)    mean-rank bars + dataset x method heatmap
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import paths  # noqa: E402

INK, MUTED, MAJ, MIN, HELD, SYN = "#1b1f24", "#6b7280", "#9ca3af", "#b42318", "#1d4ed8", "#047857"


def _ax(ax, title="", xl="", yl=""):
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.set_xlabel(xl, fontsize=8, color=MUTED)
    ax.set_ylabel(yl, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax


def _pca2(X: np.ndarray, fit: np.ndarray):
    mu, sd = fit.mean(0), fit.std(0)
    sd = np.where(sd > 0, sd, 1)
    Z = (fit - mu) / sd
    _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    W = vt[:2].T if vt.shape[0] >= 2 else np.eye(Z.shape[1], 2)
    return lambda A: ((A - mu) / sd) @ W


def _grid(n, ncol=4):
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for a in axs.ravel()[n:]:
        a.axis("off")
    return fig, axs.ravel()


def data_report(name: str, seed: int = 0, q: float = 0.2, protocol: str = "clean",
                outdir: Path | None = None, max_features: int = 12) -> list[Path]:
    from .data import registry
    from .data.fetch import load_raw
    from .data.protocol import prepare
    out = Path(outdir or paths.FIGURES) / name
    out.mkdir(parents=True, exist_ok=True)
    raw = load_raw(name)
    s = prepare(name, seed=seed, q=q, protocol=protocol)
    sch = s.schema
    files = []

    # 1 funnel
    stages = [("raw", len(raw)), ("cleaned", s.log["clean_shape"][0]),
              ("train", len(s.train_major) + len(s.train_minor_full)), ("val", len(s.val)),
              ("test", len(s.test)), ("D*_minor", len(s.train_minor_full)),
              ("D_minor (q)", len(s.train_minor)), ("to generate", s.n_target)]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh([a for a, _ in stages][::-1], [b for _, b in stages][::-1], color=MUTED)
    for i, (_, v) in enumerate(stages[::-1]):
        ax.text(v, i, f" {v}", va="center", fontsize=7, color=INK)
    _ax(ax, f"{name}: protocol funnel (seed {seed}, q={q}, {protocol})", "rows")
    files.append(out / "01_funnel.png")
    fig.savefig(files[-1], dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 2 class balance
    before = pd.concat([s.train_major, s.train_minor_full])[sch.target].value_counts()
    after = s.imbalanced_train[sch.target].value_counts().reindex(before.index).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    x = np.arange(len(before))
    ax.bar(x - .2, before.values, .4, color=MAJ, label="train before q-reduction")
    ax.bar(x + .2, after.values, .4, color=MIN, label="D_train the oversampler sees")
    ax.set_xticks(x, [str(i) for i in before.index], fontsize=7)
    ax.legend(fontsize=7, frameon=False)
    _ax(ax, f"class balance (minority = {sch.minority_label!r}, IR = {s.imbalance_ratio:.1f})",
        "", "rows")
    files.append(out / "02_class_balance.png")
    fig.savefig(files[-1], dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 3 raw vs clean per feature
    spec = registry.get(name)
    rename = s.log.get("renamed_columns", {})
    inv = {v: k for k, v in rename.items()}
    feats = sch.features[:max_features]
    fig, axs = _grid(len(feats))
    clean = pd.concat([s.train_major, s.train_minor_full, s.val, s.test])
    for a, c in zip(axs, feats):
        rc = inv.get(c, c)
        if c in sch.continuous:
            rv = pd.to_numeric(raw[rc], errors="coerce").dropna()
            a.hist(rv, bins=30, color=MAJ, alpha=.7, label="raw")
            a.hist(clean[c].astype(float), bins=30, color=HELD, alpha=.5, label="clean")
        else:
            rv = raw[rc].astype(str).value_counts().head(8)
            cv = clean[c].astype(str).value_counts().reindex(
                [str(k).strip().replace(",", ";") for k in rv.index]).fillna(0)
            xx = np.arange(len(rv))
            a.bar(xx - .2, rv.values, .4, color=MAJ)
            a.bar(xx + .2, cv.values, .4, color=HELD)
            a.set_xticks(xx, [str(k)[:8] for k in rv.index], rotation=60, fontsize=6)
        _ax(a, f"{c} ({'con' if c in sch.continuous else 'cat'})")
    axs[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{name}: raw file (grey) vs cleaned (blue); dropped: "
                 f"{s.log['dropped_columns'] + s.log['unused_columns']}, "
                 f"NaN rows: {s.log['rows_dropped_missing']}  [{spec.source['type']}]",
                 fontsize=8, x=0.01, ha="left")
    fig.tight_layout()
    files.append(out / "03_features_raw_vs_clean.png")
    fig.savefig(files[-1], dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 4 minority geometry + interpolation (the "feature engineering" the method does)
    if sch.m_con >= 2:
        from ..methods.v2_corrected import interpolate as i2
        rng = np.random.default_rng(seed)
        con = sch.continuous
        A = s.imbalanced_train[con].astype(float).to_numpy()
        P = _pca2(A, A)
        n_i = min(2000, s.n_target)
        ri = i2.interpolate(s.train_minor, sch, n_i, rng, mode="random")[0]
        ki = i2.interpolate(s.train_minor, sch, n_i, rng, mode="knn")[0]
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
        for a, (title, Z, col) in zip(axs, (("real data", None, None),
                                             ("v1 interpolation (random partner)", ri, MIN),
                                             ("v2 interpolation (kNN, z-scored)", ki, SYN))):
            a.scatter(*P(s.train_major[con].astype(float).to_numpy()).T, s=2, c=MAJ, alpha=.3,
                      label="majority")
            a.scatter(*P(s.minor_heldout[con].astype(float).to_numpy()).T, s=6, c=HELD,
                      alpha=.5, label="held-out minority")
            if Z is not None and len(Z):
                a.scatter(*P(Z.to_numpy()).T, s=2, c=col, alpha=.25, label="interpolated")
            a.scatter(*P(s.train_minor[con].astype(float).to_numpy()).T, s=14, c=INK,
                      marker="x", label="D_minor")
            _ax(a, title, "PC1", "PC2")
        axs[0].legend(fontsize=6, frameon=False, markerscale=2)
        fig.tight_layout()
        files.append(out / "04_minority_geometry.png")
        fig.savefig(files[-1], dpi=130, bbox_inches="tight")
        plt.close(fig)
    (out / "summary.json").write_text(json.dumps({**s.summary(), "log": s.log}, indent=2,
                                                 default=str))
    return files


def synth_report(run_dir: Path, max_features: int = 12) -> list[Path]:
    from .data.protocol import prepare
    from .evaluate.quality import dcr_values
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    cfg = man["config"]
    s = prepare(man["dataset"], seed=man["seed"], q=cfg.get("q", 0.2),
                protocol=cfg.get("protocol", "clean"))
    syn = pd.read_csv(run_dir / "synthetic.csv")
    for c in s.schema.categorical:
        syn[c] = syn[c].astype(str)
    sch, held = s.schema, s.minor_heldout
    files = []
    feats = sch.features[:max_features]
    fig, axs = _grid(len(feats))
    for a, c in zip(axs, feats):
        if c in sch.continuous:
            lo, hi = np.nanpercentile(np.r_[held[c].astype(float), syn[c].astype(float)], [1, 99])
            b = np.linspace(lo, hi, 30)
            a.hist(held[c].astype(float), bins=b, density=True, color=HELD, alpha=.5,
                   label="held-out real")
            a.hist(syn[c].astype(float), bins=b, density=True, color=SYN, alpha=.5,
                   label="synthetic")
        else:
            cats = held[c].astype(str).value_counts().index[:8]
            xx = np.arange(len(cats))
            a.bar(xx - .2, held[c].astype(str).value_counts(normalize=True).reindex(cats), .4,
                  color=HELD)
            a.bar(xx + .2, syn[c].astype(str).value_counts(normalize=True).reindex(cats)
                  .fillna(0), .4, color=SYN)
            a.set_xticks(xx, [str(k)[:8] for k in cats], rotation=60, fontsize=6)
        _ax(a, c)
    axs[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{man['dataset']} / {man['method']} / seed {man['seed']}: held-out minority "
                 f"(blue) vs synthetic (green)", fontsize=8, x=0.01, ha="left")
    fig.tight_layout()
    files.append(run_dir / "fig_marginals.png")
    fig.savefig(files[-1], dpi=130, bbox_inches="tight")
    plt.close(fig)

    d = dcr_values(syn, s)
    fig, axs = plt.subplots(1, 2 if sch.m_con >= 2 else 1, figsize=(10, 3.2), squeeze=False)
    axs = axs.ravel()
    axs[0].hist(d, bins=40, color=SYN)
    _ax(axs[0], f"DCR: held-out minority -> nearest synthetic (median {np.median(d):.3f})",
        "normalised distance", "count")
    if sch.m_con >= 2:
        con = sch.continuous
        A = s.imbalanced_train[con].astype(float).to_numpy()
        P = _pca2(A, A)
        axs[1].scatter(*P(s.train_major[con].astype(float).to_numpy()).T, s=2, c=MAJ, alpha=.3)
        axs[1].scatter(*P(syn[con].astype(float).to_numpy()).T, s=3, c=SYN, alpha=.3)
        axs[1].scatter(*P(held[con].astype(float).to_numpy()).T, s=8, c=HELD, alpha=.7)
        _ax(axs[1], "PCA: majority (grey), synthetic (green), held-out minority (blue)")
    fig.tight_layout()
    files.append(run_dir / "fig_dcr_pca.png")
    fig.savefig(files[-1], dpi=130, bbox_inches="tight")
    plt.close(fig)
    return files


def report_figure(res: dict) -> Path | None:
    if "error" in res:
        return None
    out = paths.RUNS / res["experiment"] / "_report"
    out.mkdir(parents=True, exist_ok=True)
    m = res["mean"]
    fig, axs = plt.subplots(1, 2, figsize=(13, max(3, .35 * len(m.columns) + 1)))
    fr = res["friedman"]
    if "mean_ranks" in fr:
        mr = pd.Series(fr["mean_ranks"]).sort_values()
        axs[0].barh(mr.index[::-1], mr.values[::-1], color=MUTED)
        axs[0].axvline(mr.iloc[0] + fr["critical_difference"], color=MIN, ls="--", lw=1)
        _ax(axs[0], f"mean rank (lower better); dashed = best + CD {fr['critical_difference']}")
    im = axs[1].imshow(m.T.to_numpy(float), aspect="auto", cmap="viridis")
    axs[1].set_yticks(range(len(m.columns)), m.columns, fontsize=7)
    axs[1].set_xticks(range(len(m.index)), m.index, rotation=60, fontsize=7)
    fig.colorbar(im, ax=axs[1], fraction=.03)
    _ax(axs[1], f"{res['metric']} per dataset")
    fig.tight_layout()
    p = out / f"fig_{res['metric']}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p
