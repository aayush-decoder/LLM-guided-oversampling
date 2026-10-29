"""Generates comparative visualizations across v1, v2, and v3 methods.

Creates:
1. 01_tstr_accuracy_f1.png       - TSTR Accuracy & Macro F1 side-by-side bar plots
2. 02_fidelity_metrics.png       - Discriminator, Inverse-KL, Density, Coverage
3. 03_privacy_tradeoff.png       - DCR mean vs Exact Copy Rate
4. 04_v3_loss_dynamics.png       - L_pos, L_neg, and Total Loss over steps for v3 runs
5. 05_radar_comparison.png       - Multi-metric radar chart comparing v1, v2, v3
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from predLLM.common.tabgen.eval.report import collect
from predLLM.common.tabgen.paths import REPORTS

OUT_DIR = REPORTS / "data_viz" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling palette
PALETTE = {
    "v1": "#3498db",   # blue
    "v2": "#e67e22",   # orange
    "v3": "#2ecc71",   # green
    "paper": "#95a5a6", # gray
}


def plot_tstr_accuracy_f1(df: pd.DataFrame, out_dir: Path):
    """Bar plots of TSTR accuracy and macro F1 per dataset."""
    if df.empty:
        return
    datasets = df["dataset"].unique()
    
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
        
        for idx, metric in enumerate(["accuracy", "f1_macro"]):
            m_sub = sub[sub["metric"] == metric].dropna(subset=["score"])
            if m_sub.empty:
                continue
            ax = axes[idx]
            
            # Aggregate over seeds
            agg = m_sub.groupby(["version", "method"])["score"].agg(["mean", "std"]).reset_index()
            agg["method_label"] = agg["version"] + ":" + agg["method"]
            agg = agg.sort_values(by=["version", "mean"], ascending=[True, False])
            
            colors = [PALETTE.get(v, "#34495e") for v in agg["version"]]
            bars = ax.barh(
                agg["method_label"],
                agg["mean"],
                xerr=agg["std"].fillna(0),
                color=colors,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.8,
                capsize=3,
            )
            
            # Reference baseline line (original data score if available)
            orig = m_sub["original"].dropna().mean() if "original" in m_sub else None
            if orig is not None and not np.isnan(orig):
                ax.axvline(orig, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Real Data Baseline ({orig:.3f})")
            
            ax.set_title(f"{ds.upper()} - TSTR {metric.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Score", fontsize=10)
            ax.grid(axis="x", linestyle=":", alpha=0.6)
            if orig is not None and not np.isnan(orig):
                ax.legend(loc="lower right", fontsize=9)
            
            # Value labels on bars
            for bar, val in zip(bars, agg["mean"]):
                ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8.5)
            ax.set_xlim(0, max(1.05, (agg["mean"].max() + 0.1) if not agg.empty else 1.0))

        plt.suptitle(f"Model Comparison on {ds.upper()}: Downstream Classification Utility (TSTR)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = out_dir / f"01_tstr_comparison_{ds}.png"
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
        print(f"Saved: {save_path}")


def plot_fidelity_metrics(df: pd.DataFrame, out_dir: Path):
    """Multi-panel comparison of fidelity metrics across methods."""
    if df.empty:
        return
    
    # We take distinct runs (one row per run)
    runs = df.drop_duplicates(subset=["version", "dataset", "method", "seed"]).copy()
    if runs.empty:
        return

    metrics_info = [
        ("discriminator", "Discriminator Score (lower = better)", False),
        ("inverse_kl", "Inverse KL Divergence (higher = better)", True),
        ("coverage", "Coverage (higher = better)", True),
        ("density", "Density (target ≈ 1.0)", True),
    ]

    for ds in runs["dataset"].unique():
        sub = runs[runs["dataset"] == ds]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, (col, title, _) in enumerate(metrics_info):
            ax = axes[i]
            if col not in sub or sub[col].dropna().empty:
                ax.axis("off")
                continue
            
            agg = sub.groupby(["version", "method"])[col].mean().reset_index()
            agg["method_label"] = agg["version"] + ":" + agg["method"]
            agg = agg.sort_values(by=["version", "method"])
            colors = [PALETTE.get(v, "#34495e") for v in agg["version"]]

            bars = ax.bar(agg["method_label"], agg[col], color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xticklabels(agg["method_label"], rotation=35, ha="right", fontsize=8.5)
            ax.grid(axis="y", linestyle=":", alpha=0.6)

            for bar, val in zip(bars, agg[col]):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        plt.suptitle(f"Fidelity & Distribution Quality Comparison on {ds.upper()}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = out_dir / f"02_fidelity_comparison_{ds}.png"
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
        print(f"Saved: {save_path}")


def plot_privacy_tradeoff(df: pd.DataFrame, out_dir: Path):
    """Scatter / bar plot showing DCR mean vs Exact Copy Rate (Privacy)."""
    runs = df.drop_duplicates(subset=["version", "dataset", "method", "seed"]).copy()
    if runs.empty or "dcr_mean" not in runs or "exact_copy_rate" not in runs:
        return

    for ds in runs["dataset"].unique():
        sub = runs[runs["dataset"] == ds]
        agg = sub.groupby(["version", "method"])[["dcr_mean", "exact_copy_rate"]].mean().reset_index()
        agg["label"] = agg["version"] + ":" + agg["method"]

        fig, ax = plt.subplots(figsize=(8, 6))
        for _, row in agg.iterrows():
            c = PALETTE.get(row["version"], "#34495e")
            ax.scatter(row["dcr_mean"], row["exact_copy_rate"], s=140, color=c, edgecolors="black", linewidth=1.2, zorder=3)
            ax.annotate(row["label"], (row["dcr_mean"], row["exact_copy_rate"]),
                        xytext=(7, 4), textcoords="offset points", fontsize=9, fontweight="medium")

        ax.set_xlabel("Mean Distance to Closest Record (DCR)  [higher = more private]", fontsize=10, fontweight="bold")
        ax.set_ylabel("Exact Copy Rate  [lower = less memorisation]", fontsize=10, fontweight="bold")
        ax.set_title(f"Privacy vs Memorization Frontier on {ds.upper()}", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        save_path = out_dir / f"03_privacy_frontier_{ds}.png"
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
        print(f"Saved: {save_path}")


def plot_v3_training_loss(runs_dir: Path, out_dir: Path):
    """Plots training dynamics for v3 models (L_pos, L_neg, total loss) from train_log.jsonl."""
    log_files = list(runs_dir.glob("v3/runs/*/*/*/train_log.jsonl")) + list(runs_dir.glob("v3/runs/*/_models/*/train_log.jsonl"))
    if not log_files:
        return

    # Group by dataset and experiment
    for log_file in log_files:
        steps, losses, pos_losses, neg_losses = [], [], [], []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    steps.append(item.get("step", len(steps)))
                    losses.append(item.get("loss", 0.0))
                    pos_losses.append(item.get("loss_pos", 0.0))
                    neg_losses.append(item.get("loss_neg", 0.0))
        except Exception:
            continue

        if not steps or len(steps) < 2:
            continue

        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        ax1.plot(steps, losses, label="Total Joint Loss (L_v3)", color="#2c3e50", linewidth=2.0)
        ax1.plot(steps, pos_losses, label="Positive Cross-Entropy (L_pos)", color="#27ae60", linewidth=1.6, linestyle="-")
        ax1.set_xlabel("Optimization Step", fontsize=10)
        ax1.set_ylabel("Positive / Total Loss", color="#2c3e50", fontsize=10)
        ax1.tick_params(axis="y", labelcolor="#2c3e50")

        # Second axis for negative unlikelihood repulsion loss
        ax2 = ax1.twinx()
        ax2.plot(steps, neg_losses, label="Negative Repulsion Loss (L_neg)", color="#e74c3c", linewidth=1.6, linestyle="--")
        ax2.set_ylabel("Negative Unlikelihood Penalty (L_neg)", color="#e74c3c", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#e74c3c")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

        dataset_name = log_file.parts[-3] if "seed" in log_file.parts[-2] else log_file.parts[-4]
        ax1.set_title(f"v3 Negative Fine-Tuning Dynamics ({dataset_name})", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        clean_name = f"{dataset_name}_{log_file.parent.name}"
        save_path = out_dir / f"04_v3_loss_dynamics_{clean_name}.png"
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
        print(f"Saved: {save_path}")


def plot_data_augmentation(df: pd.DataFrame, out_dir: Path):
    """Bar plot comparing Real Only vs Synthetic Only vs Augmented (Real + Synthetic)."""
    if "augmented" not in df.columns or df["augmented"].dropna().empty:
        return

    sub = df.dropna(subset=["augmented", "score", "original"])
    if sub.empty:
        return

    datasets = sub["dataset"].unique()
    for ds in datasets:
        ds_sub = sub[sub["dataset"] == ds]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        for idx, metric in enumerate(["accuracy", "f1_macro"]):
            m_sub = ds_sub[ds_sub["metric"] == metric]
            if m_sub.empty:
                continue
            ax = axes[idx]

            # Aggregate over seeds per method
            agg = m_sub.groupby(["version", "method"])[["original", "score", "augmented"]].mean().reset_index()
            agg["label"] = agg["version"] + ":" + agg["method"]

            y = np.arange(len(agg))
            height = 0.25

            ax.barh(y - height, agg["original"], height=height, label="1. Real Train Only", color="#95a5a6", alpha=0.9, edgecolor="black")
            ax.barh(y, agg["score"], height=height, label="2. Synthetic Only (TSTR)", color="#3498db", alpha=0.85, edgecolor="black")
            ax.barh(y + height, agg["augmented"], height=height, label="3. Real + Synthetic (Augmented)", color="#2ecc71", alpha=0.95, edgecolor="black")

            ax.set_yticks(y)
            ax.set_yticklabels(agg["label"], fontsize=9)
            ax.set_xlabel(f"{metric.replace('_', ' ').title()}", fontsize=10)
            ax.set_title(f"{ds.upper()} - Data Augmentation Comparison ({metric})", fontsize=11, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.6, axis="x")
            ax.legend(loc="lower right", fontsize=8)

        plt.tight_layout()
        save_path = out_dir / f"05_data_augmentation_{ds}.png"
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
        print(f"Saved: {save_path}")


def generate_all_graphs():
    print(f"Collecting cross-version run results from v1, v2, v3...")
    df = collect(["v1", "v2", "v3"])
    if df.empty:
        print("No run metrics found.")
        return

    print(f"Found {len(df)} metric records. Generating comparative visualizations...")
    plot_tstr_accuracy_f1(df, OUT_DIR)
    plot_fidelity_metrics(df, OUT_DIR)
    plot_privacy_tradeoff(df, OUT_DIR)
    plot_v3_training_loss(Path("."), OUT_DIR)
    plot_data_augmentation(df, OUT_DIR)
    print("All comparison graphs generated successfully!")


if __name__ == "__main__":
    generate_all_graphs()
