"""Plots for one finished run (real train vs synthetic), reproducing the paper's figures:
    marginals.png           per-feature real vs synthetic histograms
    correlation_diff.png    |corr(real) - corr(synthetic)|
    tsne.png                Fig. 8: t-SNE of real and synthetic samples coloured by class
    dcr.png                 Fig. 6: DCR histograms (real-test -> real-train vs real -> synthetic)
    feature_class_importance.png  Fig. 2: mean |SHAP| per (feature, class) from XGBoost pred_contribs
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..data.preprocess import TARGET  # noqa: E402
from ..eval.fidelity import dcr  # noqa: E402


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def marginals(real, fake, cols, path, max_cols=16):
    cols = cols[:max_cols]
    nc = min(4, len(cols))
    nr = math.ceil(len(cols) / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 2.6 * nr), squeeze=False)
    for ax, c in zip(axes.ravel(), cols):
        bins = np.histogram_bin_edges(real[c].astype(float), bins=25)
        ax.hist(real[c].astype(float), bins=bins, alpha=.55, density=True, label="real")
        ax.hist(fake[c].astype(float).clip(bins[0], bins[-1]), bins=bins, alpha=.55, density=True, label="synthetic")
        ax.set_title(str(c)[:30], fontsize=9)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8)
    _save(fig, path)


def correlation_diff(real, fake, cols, path):
    d = (real[cols].corr() - fake[cols].corr()).abs().fillna(1.0)
    fig, ax = plt.subplots(figsize=(0.35 * len(cols) + 3, 0.35 * len(cols) + 2))
    im = ax.imshow(d, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols)), cols, fontsize=7)
    ax.set_title(f"|corr(real)-corr(syn)|  mean={np.nanmean(d.values):.3f}")
    fig.colorbar(im)
    _save(fig, path)


def tsne(real, fake, feats, path, n=1000, seed=0):
    from sklearn.manifold import TSNE
    r = real.sample(min(n, len(real)), random_state=seed)
    f = fake.sample(min(n, len(fake)), random_state=seed)
    emb = TSNE(n_components=2, random_state=seed, init="pca").fit_transform(
        np.vstack([r[feats].to_numpy(float), f[feats].to_numpy(float)]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, (lab, part, df) in zip(axes, [("Real", emb[:len(r)], r), ("Synthetic", emb[len(r):], f)]):
        ax.scatter(part[:, 0], part[:, 1], c=df[TARGET].to_numpy(float), s=6, cmap="tab10")
        ax.set_title(lab)
    _save(fig, path)


def dcr_hist(split, fake, feats, path):
    tr = split.train[feats].to_numpy(float)
    orig = dcr(tr, split.test[feats].to_numpy(float))
    syn = dcr(fake[feats].to_numpy(float), tr)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    axes[0].hist(orig, bins=40, color="#4C72B0")
    axes[0].set_title("Original: test -> train")
    axes[1].hist(syn, bins=40, color="#DD8452")
    axes[1].set_title("real train -> synthetic")
    for ax in axes:
        ax.set_xlabel("DCR")
    _save(fig, path)


def feature_class_importance(real, fake, feats, n_classes, path, seed=0):
    import xgboost as xgb
    panels = []
    for df in (real, fake):
        y = df[TARGET].astype(int).to_numpy()
        if len(np.unique(y)) != n_classes:
            panels.append(None)
            continue
        clf = xgb.XGBClassifier(random_state=seed, n_jobs=-1).fit(df[feats].to_numpy(float), y)
        contrib = clf.get_booster().predict(xgb.DMatrix(real[feats].to_numpy(float)), pred_contribs=True)
        contrib = contrib.reshape(len(real), -1, len(feats) + 1) if contrib.ndim == 3 else contrib[:, None, :]
        panels.append(np.abs(contrib[..., :-1]).mean(axis=0))          # (classes, features)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.4 * max(3, n_classes) + 2))
    for ax, p, t in zip(axes, panels, ("Original", "Synthetic")):
        if p is None:
            ax.set_title(f"{t}: missing classes")
            continue
        ax.imshow(p, aspect="auto", cmap="viridis")
        ax.set_title(t)
        ax.set_xlabel("features")
        ax.set_ylabel("classes")
    _save(fig, path)


def plot_run(run_dir: Path, split, fake: pd.DataFrame) -> list[Path]:
    out = Path(run_dir) / "plots"
    out.mkdir(exist_ok=True)
    feats = split.features
    real = split.train
    marginals(real, fake, feats + [TARGET], out / "marginals.png")
    correlation_diff(real, fake, feats + [TARGET], out / "correlation_diff.png")
    dcr_hist(split, fake, feats, out / "dcr.png")
    tsne(real, fake, feats, out / "tsne.png")
    if split.task == "classification":
        feature_class_importance(real, fake, feats, split.n_classes, out / "feature_class_importance.png")
    return sorted(out.glob("*.png"))
