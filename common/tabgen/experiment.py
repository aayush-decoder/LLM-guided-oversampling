"""
Version-agnostic experiment runner (the evaluation phase of Fig. 1 of the paper).

For every seed:  split -> [fit generator] -> generate |D_fake| = gen_size*|D_real| rows
                 -> save synthetic data -> TSTR + fidelity metrics -> metrics.json
Finished seeds are skipped unless overwrite=True, so interrupted benchmarks resume where they stopped.

Run directory layout (<version>/runs/<dataset>/<method>__<llm>/):
    seed<k>/config.json  split.json  preprocessor.json
    seed<k>/model/                  fine-tuned LLM (full weights or LoRA adapter) + llm_meta.json
    seed<k>/train_log.jsonl         loss curve (LLM methods)
    seed<k>/synthetic_encoded.csv   D_fake in the encoded (evaluation) space
    seed<k>/synthetic_clean.csv     D_fake decoded to real categories / units
    seed<k>/generation_stats.json   validity rate, retries, timings
    seed<k>/metrics.json            TSTR (+ Original reference) + fidelity
    seed<k>/plots/*.png             with --plots
    summary.json                    mean/std over seeds
"""

from typing import Any, Optional, List, Dict, Union

# Placeholder for constant / export
NON_LLM: Any = None

def exp_dir(version, dataset, method, llm_tag, cfg):
    """<version>/runs/[<runs_subdir>/]<dataset>[_bin]/<method>__<llm>"""
    pass

def split_from_cfg(dataset, seed, cfg):
    pass

def evaluate(split, enc, cfg, seed):
    pass

def run_seed(version, dataset, method, seed, cfg, make_generator, llm_tag, overwrite, plots):
    pass

def summarize(results):
    pass

def run_experiment(version, dataset, method, cfg, make_generator, llm_tag, seeds, overwrite, plots):
    pass
