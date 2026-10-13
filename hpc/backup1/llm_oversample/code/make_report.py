#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final cross-dataset report for a finished (or partly finished) main.py run.

    python make_report.py --run_dir runs/exp1            # writes runs/exp1/final_report/
    python make_report.py --run_dir runs/exp1 --out my_report --datasets adult bank credit_card

Reads  runs/<run>/<dataset>/seed_*/metrics/*.json   (the single source of truth)
Writes tables (csv / markdown / latex) and charts (png) into <run_dir>/final_report/.

Roles of the methods
  Original            upper bound   (full training set, nothing removed)        -> shown, never ranked
  Imbalanced          floor         (q=0.2 reduced training set, no correction)
  RandomOverSampler, SMOTE / SMOTE-NC, ADASYN     classical benchmarks
  ImbLLM-random       ORIGINAL PAPER (random fix_y order)
  ImbLLM-IAFP         PROPOSED       (importance-aware fix_y)

SMOTE (numeric-only data) and SMOTE-NC (mixed data) are merged into one column "SMOTE*" so that it
exists for every dataset. ADASYN only exists for numeric-only data, so it is reported but not ranked.
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

METRICS = ["f1", "g_score", "pr_auc", "roc_auc", "recall", "precision", "accuracy"]
METRIC_LABEL = {"f1": "F1", "g_score": "G-score", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC",
                "recall": "Recall", "precision": "Precision", "accuracy": "Accuracy"}
HEADLINE = ["f1", "g_score", "pr_auc", "roc_auc"]

ORDER = ["Original", "Imbalanced", "RandomOverSampler", "SMOTE*", "ADASYN", "ImbLLM-random", "ImbLLM-IAFP"]
RANKED = ["Imbalanced", "RandomOverSampler", "SMOTE*", "ImbLLM-random", "ImbLLM-IAFP"]   # needs full coverage
ROLE = {"Original": "upper bound", "Imbalanced": "floor", "RandomOverSampler": "benchmark", "SMOTE*": "benchmark",
        "ADASYN": "benchmark", "ImbLLM-random": "original paper", "ImbLLM-IAFP": "PROPOSED"}
LABEL = {"ImbLLM-random": "ImbLLM-random\n(paper)", "ImbLLM-IAFP": "ImbLLM-IAFP\n(proposed)"}
COLOR = {"Original": "#8c8c8c", "Imbalanced": "#c9b037", "RandomOverSampler": "#1f77b4", "SMOTE*": "#17a2b8",
         "ADASYN": "#9467bd", "ImbLLM-random": "#ff7f0e", "ImbLLM-IAFP": "#2ca02c"}


def lab(m):
    return LABEL.get(m, m)


# ----------------------------------------------------------------------------- loading
def load(run_dir: Path, only):
    rows = []
    for dd in sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / "dataset_info.json").exists()):
        if only and dd.name not in only:
            continue
        for f in sorted(dd.glob("seed_*/metrics/*.json")):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            rows.append({k: v for k, v in r.items() if not isinstance(v, (dict, list))})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no metrics found under {run_dir}")
    df["method"] = df["method"].replace({"SMOTE": "SMOTE*", "SMOTE-NC": "SMOTE*"})
    return df


def dataset_status(run_dir: Path):
    try:
        st = json.load(open(run_dir / "status.json"))["datasets"]
        return {k: v.get("state", "?") for k, v in st.items()}
    except Exception:
        return {}


def check_same_test_set(df):
    bad = df.groupby(["dataset", "seed"])["test_fingerprint"].nunique()
    bad = bad[bad > 1]
    return [f"{d} seed {s}" for d, s in bad.index]


# ----------------------------------------------------------------------------- tables
def md_table(df, bold_best=None):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(out) + "\n"


def per_dataset_tables(df, outdir):
    g = df.groupby(["dataset", "method"])
    mean, std = g[METRICS].mean(), g[METRICS].std().fillna(0)
    nseed = g["seed"].nunique()
    tdir = outdir / "tables"
    paths = {}
    for m in METRICS:
        pm = mean[m].unstack().reindex(columns=[c for c in ORDER if c in mean.index.get_level_values(1)])
        ps = std[m].unstack().reindex(columns=pm.columns)
        pm.round(4).to_csv(tdir / f"per_dataset_{m}_mean.csv")
        ps.round(4).to_csv(tdir / f"per_dataset_{m}_std.csv")
        txt = pm.copy().astype(object)
        for d in pm.index:
            cand = pm.loc[d, [c for c in RANKED + ["ADASYN"] if c in pm.columns]].dropna()
            best = cand.idxmax() if len(cand) else None
            for c in pm.columns:
                v = pm.loc[d, c]
                if pd.isna(v):
                    txt.loc[d, c] = "–"
                else:
                    s = f"{v:.3f}±{ps.loc[d, c]:.3f}"
                    txt.loc[d, c] = f"**{s}**" if c == best else s
        txt.columns = [c.replace("\n", " ") for c in txt.columns]
        (tdir / f"per_dataset_{m}.md").write_text(
            f"Per-dataset {METRIC_LABEL[m]} (mean±std over seeds; bold = best among non-Original methods; – = not applicable / failed)\n\n"
            + md_table(txt.reset_index()), encoding="utf-8")
        paths[m] = pm
    return paths, mean


def overall_table(per, outdir):
    """per[metric] = datasets x methods (mean over seeds)."""
    rows = []
    # datasets where every ranked method exists -> fair averages / ranks
    full = per["f1"][RANKED].dropna().index
    for meth in [c for c in ORDER if c in per["f1"].columns]:
        r = {"method": meth, "role": ROLE[meth], "n_datasets": int(per["f1"][meth].notna().sum())}
        for m in METRICS:
            r[f"{m}"] = per[m][meth].mean()
        for m in HEADLINE:
            if meth in RANKED:
                r[f"rank_{m}"] = per[m].loc[full, RANKED].rank(axis=1, ascending=False).mean()[meth]
        rows.append(r)
    t = pd.DataFrame(rows)
    # gap closed: how much of (Original - Imbalanced) does a method recover (F1), only where the gap is real
    gap = (per["f1"]["Original"] - per["f1"]["Imbalanced"])
    ok = gap[gap > 0.01].index
    t["f1_gap_closed_%"] = [np.nan if meth in ("Original",) else
                            100 * ((per["f1"].loc[ok, meth] - per["f1"].loc[ok, "Imbalanced"]) / gap[ok]).mean()
                            for meth in t["method"]]
    t.round(4).to_csv(outdir / "tables" / "overall_summary.csv", index=False)

    disp = t.copy()
    for m in METRICS:
        best = disp.loc[disp["method"].isin(RANKED + ["ADASYN"]), m].max()
        disp[m] = [f"**{v:.3f}**" if abs(v - best) < 1e-12 else f"{v:.3f}" for v in t[m]]
    for m in HEADLINE:
        disp[f"rank_{m}"] = [("–" if pd.isna(v) else f"{v:.2f}") for v in t[f"rank_{m}"]]
    disp["f1_gap_closed_%"] = [("–" if pd.isna(v) else f"{v:.0f}%") for v in t["f1_gap_closed_%"]]
    cols = ["method", "role", "n_datasets"] + METRICS + [f"rank_{m}" for m in HEADLINE] + ["f1_gap_closed_%"]
    md = md_table(disp[cols])
    note = (f"\nAverages are over the datasets each method ran on (n_datasets). Ranks (1 = best) use only the "
            f"{len(full)} datasets where all of {', '.join(RANKED)} exist; Original/ADASYN are not ranked. "
            f"f1_gap_closed = share of the Original-minus-Imbalanced F1 gap recovered, over datasets where that gap > 0.01.\n")
    (outdir / "tables" / "overall_summary.md").write_text("# Overall summary\n\n" + md + note, encoding="utf-8")
    try:
        latex = t[cols].round(3).to_latex(index=False, escape=True, na_rep="--")
        (outdir / "tables" / "overall_summary.tex").write_text(latex, encoding="utf-8")
    except Exception:
        pass
    return t, full


def significance(per, full, outdir):
    from scipy.stats import wilcoxon
    rows = []
    for m in METRICS:
        P = per[m].loc[full]
        for other in [c for c in RANKED + ["ADASYN", "Original"] if c != "ImbLLM-IAFP" and c in per[m].columns]:
            d = (per[m]["ImbLLM-IAFP"] - per[m][other]).dropna()
            if len(d) == 0:
                continue
            try:
                p = wilcoxon(d).pvalue if (d.abs() > 1e-12).any() else 1.0
            except Exception:
                p = np.nan
            rows.append({"metric": m, "proposed": "ImbLLM-IAFP", "versus": other, "n_datasets": len(d),
                         "wins": int((d > 1e-9).sum()), "ties": int((d.abs() <= 1e-9).sum()),
                         "losses": int((d < -1e-9).sum()), "mean_delta": d.mean(), "median_delta": d.median(),
                         "wilcoxon_p": p})
    t = pd.DataFrame(rows)
    t.round(5).to_csv(outdir / "tables" / "proposed_vs_others_win_tie_loss.csv", index=False)
    show = t[t["metric"].isin(HEADLINE)].copy()
    show["W/T/L"] = show["wins"].astype(str) + "/" + show["ties"].astype(str) + "/" + show["losses"].astype(str)
    show["mean_delta"] = show["mean_delta"].map(lambda v: f"{v:+.4f}")
    show["wilcoxon_p"] = show["wilcoxon_p"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
    (outdir / "tables" / "proposed_vs_others_win_tie_loss.md").write_text(
        "# ImbLLM-IAFP (proposed) versus every other method\n\nW/T/L = datasets where IAFP is better/equal/worse "
        "(per-dataset mean over seeds). Wilcoxon signed-rank p-value over datasets (p < 0.05 = consistent difference).\n\n"
        + md_table(show[["metric", "versus", "n_datasets", "W/T/L", "mean_delta", "wilcoxon_p"]]), encoding="utf-8")
    return t


# ----------------------------------------------------------------------------- charts
def style(ax):
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def chart_overall(per, outdir, ndata):
    meths = [c for c in ORDER if c in per["f1"].columns]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    for ax, m in zip(axes.ravel(), METRICS):
        mu = [per[m][c].mean() for c in meths]
        se = [per[m][c].std() / np.sqrt(max(1, per[m][c].notna().sum())) for c in meths]
        ax.bar(range(len(meths)), mu, yerr=se, capsize=3, color=[COLOR[c] for c in meths], edgecolor="black", lw=0.5)
        for i, v in enumerate(mu):
            ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(meths)))
        ax.set_xticklabels([lab(c) for c in meths], rotation=40, ha="right", fontsize=8)
        lo = max(0, min(mu) - 0.12)
        ax.set_ylim(lo, min(1.0, max(mu) + 0.06))
        ax.set_title(METRIC_LABEL[m], fontweight="bold")
        style(ax)
    ax = axes.ravel()[-1]
    ax.axis("off")
    ax.text(0, 0.95, "Method roles", fontweight="bold", va="top")
    y = 0.85
    for c in meths:
        ax.add_patch(plt.Rectangle((0, y - 0.03), 0.06, 0.06, color=COLOR[c], transform=ax.transAxes))
        ax.text(0.1, y, f"{c}  —  {ROLE[c]}", va="center", fontsize=9, transform=ax.transAxes)
        y -= 0.1
    ax.text(0, 0.05, "bars = mean over datasets\nerror bars = standard error across datasets\n(y-axis does not start at 0)",
            fontsize=8, color="#555", transform=ax.transAxes)
    fig.suptitle(f"Mean metric across {ndata} datasets (same held-out test set for every method)", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "charts" / "01_overall_metrics.png", dpi=170)
    plt.close(fig)


def chart_per_dataset(per, outdir):
    meths = [c for c in ORDER if c in per["f1"].columns]
    for m in METRICS:
        P = per[m][meths]
        P = P.loc[P["Original"].sort_values(ascending=False).index] if "Original" in P else P
        n, k = len(P), len(meths)
        w = 0.8 / k
        fig, ax = plt.subplots(figsize=(max(12, 0.9 * n + 4), 5.5))
        for j, c in enumerate(meths):
            ax.bar(np.arange(n) + (j - k / 2 + 0.5) * w, P[c].to_numpy(float), w, label=c.replace("\n", " "),
                   color=COLOR[c], edgecolor="black", lw=0.3)
        ax.set_xticks(range(n))
        ax.set_xticklabels(P.index, rotation=45, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(METRIC_LABEL[m])
        ax.set_title(f"{METRIC_LABEL[m]} per dataset (mean over seeds)")
        ax.legend(ncol=4, fontsize=8, loc="lower left")
        style(ax)
        fig.tight_layout()
        fig.savefig(outdir / "charts" / f"02_per_dataset_{m}.png", dpi=150)
        plt.close(fig)


def chart_heatmap(per, outdir, metric="f1"):
    """Improvement over the Imbalanced floor, per dataset, per method."""
    meths = [c for c in ORDER if c in per[metric].columns and c not in ("Imbalanced",)]
    D = per[metric][meths].sub(per[metric]["Imbalanced"], axis=0)
    D = D.loc[D["Original"].sort_values(ascending=False).index] if "Original" in D else D
    v = max(0.05, np.nanpercentile(np.abs(D.to_numpy(float)), 95))
    fig, ax = plt.subplots(figsize=(1.3 * len(meths) + 3, 0.38 * len(D) + 2.5))
    im = ax.imshow(D.to_numpy(float), cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(meths)))
    ax.set_xticklabels([lab(c) for c in meths], rotation=30, ha="right")
    ax.set_yticks(range(len(D)))
    ax.set_yticklabels(D.index)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            x = D.iloc[i, j]
            if np.isfinite(x):
                ax.text(j, i, f"{x:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"{METRIC_LABEL[metric]} gain over Imbalanced baseline (blue = better, red = worse)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(outdir / "charts" / f"03_gain_over_imbalanced_{metric}.png", dpi=150)
    plt.close(fig)


def chart_rank(t, full, outdir):
    fig, axes = plt.subplots(1, len(HEADLINE), figsize=(5 * len(HEADLINE), 4.4), sharey=True)
    tt = t[t["method"].isin(RANKED)]
    for ax, m in zip(axes, HEADLINE):
        vals = tt.set_index("method")[f"rank_{m}"]
        ax.bar(range(len(vals)), vals.values, color=[COLOR[c] for c in vals.index], edgecolor="black", lw=0.5)
        for i, v in enumerate(vals.values):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([lab(c) for c in vals.index], rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{METRIC_LABEL[m]}")
        style(ax)
    axes[0].set_ylabel("average rank (1 = best)")
    fig.suptitle(f"Average rank over {len(full)} datasets (lower is better)")
    fig.tight_layout()
    fig.savefig(outdir / "charts" / "04_average_rank.png", dpi=160)
    plt.close(fig)


def chart_proposed_vs_paper(per, outdir):
    fig, axes = plt.subplots(1, len(HEADLINE), figsize=(5.2 * len(HEADLINE), 0.3 * len(per["f1"]) + 3), sharey=True)
    for ax, m in zip(axes, HEADLINE):
        d = (per[m]["ImbLLM-IAFP"] - per[m]["ImbLLM-random"]).dropna().sort_values()
        ax.barh(d.index, d.values, color=["#2ca02c" if v > 0 else "#d62728" for v in d.values])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{METRIC_LABEL[m]}\nmean Δ={d.mean():+.4f}  (IAFP better on {int((d > 0).sum())}/{len(d)})", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Proposed ImbLLM-IAFP minus original-paper ImbLLM-random, per dataset (green = proposed better)")
    fig.tight_layout()
    fig.savefig(outdir / "charts" / "05_proposed_vs_paper.png", dpi=160)
    plt.close(fig)


def chart_box(per, outdir):
    meths = [c for c in ORDER if c in per["f1"].columns]
    fig, axes = plt.subplots(1, len(HEADLINE), figsize=(5.2 * len(HEADLINE), 4.8), sharey=False)
    for ax, m in zip(axes, HEADLINE):
        data = [per[m][c].dropna().to_numpy() for c in meths]
        bp = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.6)
        for patch, c in zip(bp["boxes"], meths):
            patch.set_facecolor(COLOR[c])
            patch.set_alpha(0.8)
        ax.set_xticklabels([lab(c) for c in meths], rotation=40, ha="right", fontsize=8)
        ax.set_title(METRIC_LABEL[m])
        style(ax)
    fig.suptitle("Distribution of per-dataset scores (box = quartiles, triangle = mean)")
    fig.tight_layout()
    fig.savefig(outdir / "charts" / "06_distribution_boxplots.png", dpi=160)
    plt.close(fig)


def chart_precision_recall(per, outdir):
    meths = [c for c in ORDER if c in per["f1"].columns]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for c in meths:
        ax.scatter(per["recall"][c].mean(), per["precision"][c].mean(), s=130, color=COLOR[c], edgecolor="black",
                   label=c, zorder=3)
    ax.set_xlabel("mean recall")
    ax.set_ylabel("mean precision")
    ax.set_title("Precision/recall trade-off (mean over datasets)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "charts" / "07_precision_vs_recall.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs/exp1")
    ap.add_argument("--out", default=None, help="default: <run_dir>/final_report")
    ap.add_argument("--datasets", nargs="*", default=None)
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    outdir = Path(a.out) if a.out else run_dir / "final_report"
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "charts").mkdir(parents=True, exist_ok=True)

    df = load(run_dir, set(a.datasets or []))
    status = dataset_status(run_dir)
    bad = check_same_test_set(df)
    print(f"datasets with metrics: {df['dataset'].nunique()} | rows: {len(df)}")
    print("SAME-TEST-SET CHECK:", "OK (one test fingerprint per dataset+seed)" if not bad else f"MISMATCH in {bad}")

    per, mean = per_dataset_tables(df, outdir)
    for need in ["Original", "Imbalanced", "ImbLLM-random", "ImbLLM-IAFP"]:
        if need not in per["f1"].columns:
            raise SystemExit(f"method {need} missing from results; cannot build report")
    for c in RANKED:
        if c not in per["f1"].columns:
            per = {m: p.assign(**{c: np.nan}) for m, p in per.items()}
    t, full = overall_table(per, outdir)
    sig = significance(per, full, outdir)

    cov = df.groupby(["dataset", "method"])["seed"].nunique().unstack().reindex(columns=[c for c in ORDER if c in df["method"].unique()])
    cov.insert(0, "run_status", [status.get(d, "?") for d in cov.index])
    cov.to_csv(outdir / "tables" / "coverage_seeds_per_method.csv")
    (outdir / "tables" / "coverage_seeds_per_method.md").write_text(
        "Number of seeds with metrics per dataset/method (NaN = method not applicable or failed)\n\n"
        + md_table(cov.fillna("–").reset_index()), encoding="utf-8")

    chart_overall(per, outdir, df["dataset"].nunique())
    chart_per_dataset(per, outdir)
    chart_heatmap(per, outdir, "f1")
    chart_heatmap(per, outdir, "g_score")
    chart_rank(t, full, outdir)
    chart_proposed_vs_paper(per, outdir)
    chart_box(per, outdir)
    chart_precision_recall(per, outdir)

    print("\n" + (outdir / "tables" / "overall_summary.md").read_text())
    s = sig[(sig["versus"] == "ImbLLM-random") & (sig["metric"].isin(HEADLINE))]
    print("Proposed (IAFP) vs original-paper (random):")
    print(s[["metric", "wins", "ties", "losses", "mean_delta", "wilcoxon_p"]].round(4).to_string(index=False))
    print(f"\nreport written to {outdir.resolve()}")


if __name__ == "__main__":
    main()