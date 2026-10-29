"""Aggregate one experiment: datasets x methods table, failures, Friedman/Nemenyi,
paired tests vs an anchor with Holm correction. Writes runs/<exp>/_report/.

    metric      any numeric key in metrics.json: f1 (paper), real_f1 (with the real
                minority rows), auc, auprc, f1_tuned, macro_f1, heldout_close_probability...
    anchor      method every other method is compared to (default per experiment)
    `original`  is excluded from rank tests by default: it is an upper bound that
                sees 5x the minority data, not a competitor.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import paths
from ..core.evaluate import stats
from ..core.runs import collect
from .grids import ANCHORS


def build(experiment: str, metric: str = "f1", anchor: str | None = None,
          include_original: bool = False) -> dict:
    df = collect(experiment)
    if df.empty:
        return {"error": f"no runs under {paths.RUNS / experiment}"}
    status = df.groupby("method")["status"].value_counts().unstack(fill_value=0)
    ok = df[df["status"] == "ok"]
    if metric not in ok.columns:
        return {"error": f"metric {metric!r} not found; have "
                         f"{sorted(c for c in ok.columns if c not in ('dataset', 'method', 'seed'))}"}
    methods = sorted(df["method"].unique())
    mean = ok.pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean")
    std = ok.pivot_table(index="dataset", columns="method", values=metric, aggfunc="std")
    nseed = ok.pivot_table(index="dataset", columns="method", values=metric, aggfunc="count")
    mean = mean.reindex(columns=methods)
    # every dataset that any method ran on; a missing cell = failure -> worst rank
    mean = mean.reindex(sorted(df["dataset"].unique()))
    rank_tab = mean.drop(columns=[c for c in mean.columns
                                  if c == "original" and not include_original])
    out: dict = {"experiment": experiment, "metric": metric, "mean": mean, "std": std,
                 "n_seeds": nseed, "status": status}
    out["friedman"] = stats.friedman_nemenyi(rank_tab)
    anchor = anchor or ANCHORS.get(experiment)
    if anchor and anchor in mean.columns:
        ref = mean[anchor].dropna().to_dict()
        pr = {c: stats.paired(mean[c].dropna().to_dict(), ref)
              for c in mean.columns if c != anchor}
        pr = {c: r for c, r in pr.items() if "error" not in r}
        adj = stats.holm({c: r["wilcoxon_p"] for c, r in pr.items()})
        for c in pr:
            pr[c]["holm_p"] = adj[c]
        out["paired"] = pr
        out["anchor"] = anchor
        out["covariates"] = _covariate_links(ok, mean, anchor)
    cost_cols = [c for c in ok.columns if c.startswith("cost_")]
    if cost_cols:
        out["cost"] = ok.groupby("method")[cost_cols].mean()
    return out


def _covariate_links(ok: pd.DataFrame, mean: pd.DataFrame, anchor: str) -> pd.DataFrame | None:
    """Spearman (over datasets) of each method's gain over the anchor with each
    dataset covariate: are the "helps on multimodal / high-IR / categorical-heavy
    data" predictions in what_corrected.md borne out? (review G8)"""
    from scipy.stats import spearmanr
    covs = [c for c in ok.columns if c.startswith("cov_")]
    if not covs:
        return None
    cov = ok.groupby("dataset")[covs].mean()
    rows = {}
    for c in mean.columns:
        if c == anchor:
            continue
        gain = (mean[c] - mean[anchor]).dropna()
        d = gain.index.intersection(cov.index)
        if len(d) < 5:
            continue
        rows[c] = {v: float(spearmanr(gain[d], cov.loc[d, v]).statistic)
                   if cov.loc[d, v].nunique() > 1 else float("nan") for v in covs}
    return pd.DataFrame(rows).T if rows else None


def render(res: dict) -> str:
    if "error" in res:
        return res["error"]
    m, s = res["mean"], res["std"]
    cell = lambda a, b: "" if pd.isna(a) else (f"{a:.4f}" if pd.isna(b) else f"{a:.4f}±{b:.3f}")  # noqa: E731
    tab = m.copy().astype(object)
    for c in m.columns:
        for d in m.index:
            tab.loc[d, c] = cell(m.loc[d, c], s.loc[d, c] if (d in s.index and c in s.columns)
                                 else np.nan)
    try:
        table_txt = tab.to_markdown()
    except ImportError:                       # `tabulate` not installed
        table_txt = "```\n" + tab.to_string() + "\n```"
    lines = [f"# {res['experiment']} -- metric `{res['metric']}` (mean ± std over seeds)", "",
             table_txt, "",
             "## Average over datasets (failures excluded here, ranked last below)", "",
             m.mean().sort_values(ascending=False).round(4).to_string(), "",
             "## Run status", "", res["status"].to_string(), "",
             "## Friedman + Nemenyi", "", "```", stats.cd_text(res["friedman"]), "```"]
    if "paired" in res:
        lines += ["", f"## Paired vs `{res['anchor']}` (Wilcoxon over datasets, Holm-adjusted)",
                  "", "| method | mean diff | 95% CI | wins | p | Holm p | min attainable p |",
                  "|---|---|---|---|---|---|---|"]
        for c, r in sorted(res["paired"].items(), key=lambda kv: -kv[1]["mean_diff"]):
            lines.append(f"| {c} | {r['mean_diff']:+.4f} | [{r['ci95'][0]:+.4f}, "
                         f"{r['ci95'][1]:+.4f}] | {r['wins']}/{r['n']} | {r['wilcoxon_p']:.4f} "
                         f"| {r['holm_p']:.4f} | {r['min_attainable_p']:.4f} |")
        lines.append("\nWith N datasets the smallest attainable two-sided Wilcoxon p is "
                     "2 * 0.5^N: N=3 can never reach 0.05; N>=6 is the minimum.")
    cv = res.get("covariates")
    if cv is not None and len(cv):
        lines += ["", f"## Gain over `{res['anchor']}` vs dataset covariates "
                  "(Spearman over datasets; needs >= 5 datasets)", "", "```",
                  cv.round(2).to_string(), "```"]
    if "cost" in res:
        lines += ["", "## Cost per run (mean over datasets x seeds)", "", "```",
                  res["cost"].round(2).to_string(), "```"]
    if any(str(c).endswith("_style") for c in res["mean"].columns):
        lines += ["", "Note: `*_style` methods are re-implementations of GReaT / TapTap / "
                  "Pred-LLM on this repository's engine, not the official packages."]
    return "\n".join(lines)


def write(res: dict) -> str:
    out = paths.RUNS / res["experiment"] / "_report"
    out.mkdir(parents=True, exist_ok=True)
    res["mean"].to_csv(out / f"{res['metric']}_mean.csv")
    res["std"].to_csv(out / f"{res['metric']}_std.csv")
    text = render(res)
    (out / f"report_{res['metric']}.md").write_text(text, encoding="utf-8")
    (out / f"friedman_{res['metric']}.json").write_text(json.dumps(res["friedman"], indent=2,
                                                                   default=str))
    return str(out)
