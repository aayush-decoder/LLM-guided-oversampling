"""Run bookkeeping: run directories, manifests, resume, and the model cache.

LAYOUT
    runs/<experiment>/<dataset>/<method>/seed<k>/
        manifest.json     config, config_hash, env, git sha + dirty flag, timings,
                          token counts, status ("ok" | "error" | "running")
        metrics.json      downstream (paper + with_real) and quality metrics
        diagnostics.json  corpus composition, prompts, interpolation geometry,
                          tau-bridge KL (measured from draws), order sensitivity...
        parse_stats.json  how much LLM output was unusable (yield = parsed/sampled)
        train_log.json    per-epoch loss / tokens / seconds
        splits.json       sizes, imbalance ratio, thresholds
        synthetic.csv     the generated minority rows
    runs/_models/<fingerprint>/   cached fine-tuned models (LoRA adapters are small)

RESUME / STALENESS (palt2 kept stale metrics.json from earlier runs)
    Before a run starts the directory is emptied and manifest.status="running"
    is written. `collect` reads a run only if status == "ok" AND its
    config_hash matches the manifest -- so an interrupted or failed rerun can
    never be reported with an older run's numbers. `--resume` skips items
    whose finished manifest already has the same config_hash.
"""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import paths
from ..config import stable_hash


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git() -> dict:
    def run(*a):
        return subprocess.check_output(["git", "-C", str(paths.ROOT), *a],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    try:
        sha = run("rev-parse", "--short", "HEAD")
        dirty = bool(run("status", "--porcelain", "--", "."))
        return {"git_sha": sha, "git_dirty": dirty}
    except Exception:
        return {"git_sha": None, "git_dirty": None}


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(), **_git()}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "imblearn", "torch",
                "transformers", "peft", "bitsandbytes", "sdv"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        env["gpu"] = None
    return env


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


@dataclass
class RunContext:
    experiment: str
    dataset: str
    method: str
    seed: int
    config: dict = field(default_factory=dict)
    root: Path = field(default_factory=lambda: paths.RUNS)

    @property
    def dir(self) -> Path:
        return self.root / self.experiment / self.dataset / self.method / f"seed{self.seed}"

    @property
    def config_hash(self) -> str:
        return stable_hash(self.config)

    def finished(self) -> bool:
        m = self.dir / "manifest.json"
        if not m.exists():
            return False
        d = json.loads(m.read_text())
        return d.get("status") == "ok" and d.get("config_hash") == self.config_hash

    def __enter__(self):
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        self._t0 = time.time()
        self._extra: dict = {}
        self._manifest("running")
        set_seed(self.seed)
        return self

    def note(self, **kw) -> None:
        self._extra.update(kw)

    def _manifest(self, status: str, error: str | None = None) -> None:
        write_json(self.dir / "manifest.json", {
            "experiment": self.experiment, "dataset": self.dataset, "method": self.method,
            "seed": self.seed, "status": status, "error": error,
            "config": self.config, "config_hash": self.config_hash,
            "elapsed_sec": round(time.time() - self._t0, 2),
            "environment": environment(), **getattr(self, "_extra", {})})

    def __exit__(self, exc_type, exc, tb):
        self._manifest("error" if exc_type else "ok", repr(exc) if exc else None)
        return False

    def write(self, name: str, obj) -> None:
        write_json(self.dir / name, obj)

    def write_csv(self, name: str, df) -> None:
        df.to_csv(self.dir / name, index=False)

    def elapsed(self) -> float:
        return round(time.time() - self._t0, 2)


# --------------------------------------------------------------------------- collect

def collect(experiment: str, root: Path | None = None):
    """Tidy frame of every FINISHED run of an experiment (status ok, hash matches)."""
    import pandas as pd
    base = (root or paths.RUNS) / experiment
    rows = []
    for mf in sorted(base.glob("*/*/seed*/manifest.json")):
        man = json.loads(mf.read_text())
        rec = {"dataset": man["dataset"], "method": man["method"], "seed": man["seed"],
               "status": man["status"]}
        met = mf.parent / "metrics.json"
        if man["status"] == "ok" and man.get("config_hash") == stable_hash(man["config"]) \
                and met.exists():
            m = json.loads(met.read_text())
            rec.update({k: v for k, v in m.items() if isinstance(v, (int, float))
                        and not isinstance(v, bool)})
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- model cache

class ModelCache:
    """Fine-tuned models keyed by what determines the weights: backbone, init,
    fine-tuning mode, the exact corpus text, the training config and the seed.
    Sampling-only knobs (condition, tau, temperature, Gibbs sweeps, verifier)
    are NOT in the key, so sweeping them re-uses one fine-tune."""

    def __init__(self, root: Path | None = None, enabled: bool = True):
        self.root = Path(root or paths.MODEL_CACHE)
        self.enabled = enabled

    @staticmethod
    def fingerprint(*, model: str, init: str, mode: str, corpus: list[str], train: dict,
                    seed: int) -> str:
        return stable_hash({"model": model, "init": init, "mode": mode, "seed": seed,
                            "train": train, "corpus": stable_hash(corpus, 40),
                            "format": "imbllm-v3-serialise-1"}, 16)

    def path(self, fp: str) -> Path:
        return self.root / fp

    def has(self, fp: str) -> bool:
        return self.enabled and (self.path(fp) / "engine.json").exists()

    def clear(self) -> int:
        n = sum(1 for _ in self.root.glob("*/engine.json")) if self.root.exists() else 0
        if self.root.exists():
            shutil.rmtree(self.root)
        return n


def shard(items: list, spec: str | None) -> list:
    """'i/N' -> every N-th item starting at i (one process per GPU on a multi-GPU node)."""
    if not spec:
        return items
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError("--shard must be i/N with 0 <= i < N")
    return items[i::n]


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")
