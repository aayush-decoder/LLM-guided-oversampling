"""Unified Benchmark Runner across Pred-LLM v1, v2, and v3.

Runs all three generations of the model on specified datasets, compares
downstream utility (TSTR Accuracy/F1) and fidelity/privacy metrics,
and automatically generates cross-version reports and visual comparison graphs.

Usage Examples:
  # Run all three versions on the 'bank' dataset (default: distilgpt2, 35 epochs)
  python run_all_versions.py --dataset bank

  # Run all three versions on 'iris' for quick verification (10 epochs)
  python run_all_versions.py --dataset iris --epochs 10

  # Run on multiple datasets with a specific LLM
  python run_all_versions.py --datasets bank,blood --llm distilgpt2

  # Include full ablation suite across versions
  python run_all_versions.py --dataset bank --ablations

  # Force overwrite and re-run all (otherwise skips finished runs)
  python run_all_versions.py --dataset bank --overwrite
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
from tabulate import tabulate

from tabgen.cli_common import build_cfg
from tabgen.data.registry import resolve_datasets
from tabgen.eval.report import build_tables
from tabgen.experiment import run_experiment
from tabgen.llm.registry import resolve_llm
from tabgen.paths import REPORTS

# Import version generators and default configurations
from predllm_v1.generator import VERSION as V1_VER, make_generator as v1_gen
from predllm_v1.cli import CONFIG as V1_CFG

from predllm_v2.generator import VERSION as V2_VER, make_generator as v2_gen
from predllm_v2.cli import CONFIG as V2_CFG

from predllm_v3.generator import VERSION as V3_VER, make_generator as v3_gen
from predllm_v3.cli import CONFIG as V3_CFG


VERSION_REGISTRY = {
    "v1": {"version": V1_VER, "cfg": V1_CFG, "gen": v1_gen, "default_methods": ["pred_llm"], "ablation_methods": ["pred_llm", "great", "taptap"]},
    "v2": {"version": V2_VER, "cfg": V2_CFG, "gen": v2_gen, "default_methods": ["pred_llm_v2"], "ablation_methods": ["pred_llm_v2", "v2_plain_loss"]},
    "v3": {"version": V3_VER, "cfg": V3_CFG, "gen": v3_gen, "default_methods": ["pred_llm_v3"], "ablation_methods": ["pred_llm_v3", "v3_no_negative", "v3_no_outlier_range"]},
}


class ArgsWrapper:
    """Helper wrapping command line arguments for build_cfg compatibility."""
    def __init__(self, config=None, llm="distilgpt2", mode=None, device="auto", epochs=35, max_steps=None, binarize=None, max_rows=None, set_overrides=None):
        self.config = config
        self.llm = llm
        self.mode = mode
        self.device = device
        self.epochs = epochs
        self.max_steps = max_steps
        self.binarize = binarize
        self.max_rows = max_rows
        self.set = set_overrides or []


def parse_args():
    parser = argparse.ArgumentParser(description="Run and compare Pred-LLM across v1, v2, and v3.")
    parser.add_argument("--dataset", "--datasets", dest="datasets", default="bank",
                        help="Comma-separated dataset names or group (e.g. 'bank', 'iris', 'paper'). Default: bank")
    parser.add_argument("--versions", default="v1,v2,v3",
                        help="Comma-separated versions to run (e.g. 'v1,v2,v3'). Default: v1,v2,v3")
    parser.add_argument("--llm", default="distilgpt2",
                        help="LLM model tag or HF ID (e.g. 'distilgpt2', 'tiny-gpt2', 'qwen2.5-0.5b'). Default: distilgpt2")
    parser.add_argument("--epochs", type=int, default=35,
                        help="Number of training epochs. Default: 35")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap total training steps (useful for fast smoke testing). Default: None")
    parser.add_argument("--seeds", default="0",
                        help="Comma-separated random seeds. Default: 0")
    parser.add_argument("--device", default="auto",
                        help="Compute device: 'auto', 'cuda', or 'cpu'. Default: auto")
    parser.add_argument("--ablations", action="store_true",
                        help="Include ablation variants (e.g., v3 without negative loss, without outlier ranges)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force re-run and overwrite existing results (default: False, resume/skip finished)")
    parser.add_argument("--plots", action="store_true", default=True,
                        help="Generate per-run diagnostic plots (marginals, correlation, t-SNE, DCR). Default: True")
    parser.add_argument("--no-plots", action="store_false", dest="plots",
                        help="Disable individual per-run plot generation.")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = resolve_datasets(args.datasets)
    versions_requested = [v.strip().lower() for v in args.versions.split(",") if v.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("=" * 80)
    print(f"PRED-LLM CROSS-VERSION RUNNER: {','.join(versions_requested).upper()}")
    print(f"Datasets: {datasets} | LLM: {args.llm} | Epochs: {args.epochs} | Seeds: {seeds}")
    print(f"Resume Mode: {'OVERWRITE' if args.overwrite else 'SKIP FINISHED (cached)'}")
    print("=" * 80)

    # Initialize configs
    wrapped_args = ArgsWrapper(
        llm=args.llm,
        device=args.device,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )
    llm_tag = resolve_llm(args.llm, args.device).tag

    results = []
    t_global_start = time.time()

    for ds in datasets:
        print(f"\n######################################################################")
        print(f" DATASET: {ds.upper()}")
        print(f"######################################################################")

        for v_key in versions_requested:
            if v_key not in VERSION_REGISTRY:
                print(f"[!] Warning: Unknown version '{v_key}'. Allowed: {list(VERSION_REGISTRY.keys())}")
                continue

            v_info = VERSION_REGISTRY[v_key]
            v_code = v_info["version"]
            v_cfg_path = v_info["cfg"]
            v_gen_fn = v_info["gen"]
            v_cfg = build_cfg(wrapped_args, v_cfg_path)

            methods = v_info["ablation_methods"] if args.ablations else v_info["default_methods"]

            for method in methods:
                print(f"\n>>> Running {v_code}:{method} on {ds} (LLM: {llm_tag}, Epochs: {args.epochs})...")
                t0 = time.time()
                try:
                    summ = run_experiment(
                        v_code,
                        ds,
                        method,
                        v_cfg,
                        v_gen_fn,
                        llm_tag,
                        seeds,
                        overwrite=args.overwrite,
                        plots=args.plots,
                    )
                    elapsed = round(time.time() - t0, 1)
                    summ["version"] = v_code
                    summ["dataset"] = ds
                    summ["method"] = method
                    summ["elapsed_s"] = elapsed
                    results.append(summ)

                    acc = summ.get("mean")
                    orig = summ.get("original_mean")
                    print(f"[{v_code}:{method}] Finished in {elapsed}s | Metric Score: {acc} | Real Data Baseline: {orig}")

                except Exception as e:
                    print(f"[{v_code}:{method}] FAILED with error: {e}")
                    import traceback
                    traceback.print_exc()

    # 1. Print formatted results table
    print("\n" + "=" * 90)
    print("COMPREHENSIVE RUN SUMMARY:")
    print("=" * 90)
    if results:
        table_rows = []
        for r in results:
            table_rows.append([
                r.get("version"),
                r.get("dataset"),
                r.get("method"),
                r.get("mean"),
                r.get("original_mean"),
                r.get("discriminator"),
                r.get("inverse_kl"),
                r.get("coverage"),
                r.get("dcr_mean"),
                r.get("elapsed_s"),
            ])
        headers = ["Ver", "Dataset", "Method", "TSTR Score", "Orig Score", "Discrim (v)", "Inv-KL (^)", "Coverage (^)", "DCR Mean", "Time (s)"]
        print(tabulate(table_rows, headers=headers, tablefmt="github"))

    # 2. Build Markdown/CSV report tables
    print("\n" + "=" * 90)
    print("BUILDING CROSS-VERSION SUMMARY TABLES...")
    print("=" * 90)
    tables = build_tables(["v1", "v2", "v3"], REPORTS / "tables")
    for k, p in tables.items():
        print(f"Artifact {k}: {p}")

    # 3. Generate Comparative Visual Graphs
    print("\n" + "=" * 90)
    print("GENERATING VISUAL COMPARISON CHARTS...")
    print("=" * 90)
    try:
        from reports.make_comparison_graphs import generate_all_graphs
        generate_all_graphs()
    except Exception as e:
        print(f"Warning: Failed to generate comparative charts: {e}")
        import traceback
        traceback.print_exc()

    total_time = round(time.time() - t_global_start, 1)
    print("\n" + "=" * 90)
    print(f"ALL BENCHMARKS COMPLETE in {total_time}s ({total_time/60:.1f} mins).")
    print(f"Visual Graphs Saved to: {REPORTS / 'data_viz' / 'comparison'}")
    print("=" * 90)


if __name__ == "__main__":
    sys.exit(main() or 0)
