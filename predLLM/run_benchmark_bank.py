"""Benchmark script to compare v2 and v3 on the 'bank' dataset (35 epochs).

Evaluates:
1. v2:pred_llm_v2 (Standard schema-constrained generator)
2. v3:pred_llm_v3 (Full v3: Negative Unlikelihood + Outlier Range + Anomaly Gating)
3. v3:v3_no_negative (Ablation: Outlier range only, no negative repulsion)
4. v3:v3_no_outlier_range (Ablation: Negative unlikelihood only, global range)

Metrics computed:
- TSTR Macro-F1 (for imbalanced minority performance)
- TSTR Accuracy
- TSTR Augmented Macro-F1 (using synthetic data for data augmentation)
- Fidelity metrics: Discriminator AUC/score, Inverse KL, Density, Coverage, DCR (privacy)
"""
import sys
import time
from pathlib import Path
import pandas as pd
from tabulate import tabulate

from predLLM.common.tabgen.cli_common import build_cfg
from predLLM.common.tabgen.eval.report import build_tables
from predLLM.common.tabgen.experiment import run_experiment
from predLLM.common.tabgen.llm.registry import resolve_llm
from predLLM.common.tabgen.paths import REPORTS

from predLLM.v2.predllm_v2.generator import VERSION as V2_VER, make_generator as v2_gen
from predLLM.v2.predllm_v2.cli import CONFIG as V2_CFG
from predLLM.v3.predllm_v3.generator import VERSION as V3_VER, make_generator as v3_gen
from predLLM.v3.predllm_v3.cli import CONFIG as V3_CFG

DATASET = "bank"
EPOCHS = 35
SEEDS = [0]
LLM = "distilgpt2"


class DummyArgs:
    def __init__(self, epochs=35, llm="distilgpt2"):
        self.config = None
        self.llm = llm
        self.mode = None
        self.device = "auto"
        self.epochs = epochs
        self.max_steps = None
        self.binarize = None
        self.max_rows = None
        self.set = []


def run_all():
    print("=" * 70)
    print(f"BENCHMARK: {DATASET.upper()} DATASET | {EPOCHS} EPOCHS | {LLM} ON GPU")
    print("=" * 70)

    args = DummyArgs(epochs=EPOCHS, llm=LLM)
    v2_config = build_cfg(args, V2_CFG)
    v3_config = build_cfg(args, V3_CFG)
    tag = resolve_llm(LLM, "auto").tag

    experiments = [
        ("v2", V2_VER, "pred_llm_v2", v2_config, v2_gen),
        ("v3", V3_VER, "pred_llm_v3", v3_config, v3_gen),
        ("v3", V3_VER, "v3_no_negative", v3_config, v3_gen),
        ("v3", V3_VER, "v3_no_outlier_range", v3_config, v3_gen),
    ]

    results = []
    t_start = time.time()

    for label, version, method, cfg, gen_fn in experiments:
        print(f"\n>>> Running {version}:{method} on {DATASET} ({EPOCHS} epochs, seed {SEEDS[0]})...")
        t0 = time.time()
        try:
            summ = run_experiment(
                version,
                DATASET,
                method,
                cfg,
                gen_fn,
                tag,
                SEEDS,
                overwrite=False,
                plots=True,
            )
            elapsed = round(time.time() - t0, 1)
            summ["elapsed_s"] = elapsed
            results.append(summ)
            print(f"[{method}] Completed in {elapsed}s | Accuracy: {summ.get('mean')} | Original: {summ.get('original_mean')}")
        except Exception as e:
            print(f"[{method}] Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("BANK BENCHMARK RESULTS SUMMARY:")
    print("=" * 70)
    if results:
        table_rows = []
        for r in results:
            table_rows.append([
                r.get("method"),
                r.get("mean"),
                r.get("original_mean"),
                r.get("discriminator"),
                r.get("inverse_kl"),
                r.get("coverage"),
                r.get("dcr_mean"),
                r.get("elapsed_s"),
            ])
        headers = ["Method", "TSTR Acc", "Orig Acc", "Discrim", "Inv-KL", "Coverage", "DCR Mean", "Time (s)"]
        print(tabulate(table_rows, headers=headers, tablefmt="github"))

    print("\n" + "=" * 70)
    print("GENERATING CROSS-VERSION COMPARISON REPORTS...")
    print("=" * 70)
    tables = build_tables(["v1", "v2", "v3"], REPORTS / "tables")
    for k, p in tables.items():
        print(f"Artifact {k}: {p}")

    try:
        from predLLM.reports.make_comparison_graphs import generate_all_graphs
        print("\n" + "=" * 70)
        print("GENERATING VISUAL COMPARISON GRAPHS...")
        print("=" * 70)
        generate_all_graphs()
    except Exception as e:
        print(f"Warning: could not generate comparison graphs: {e}")

    total_time = round(time.time() - t_start, 1)
    print(f"\nAll experiments finished in {total_time}s ({total_time/60:.1f} mins).")


if __name__ == "__main__":
    sys.exit(run_all() or 0)
