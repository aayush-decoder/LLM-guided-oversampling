"""Downloads each dataset once into data/raw/<name>.csv, plus <name>.meta.json holding the target
and the column types.

Order of attempts: sklearn/openml/url auto-download -> public mirrors -> Kaggle API -> manual
(see docs/DATASETS.md). A manually placed file may use either the registry name (gem_price.csv)
or the original Kaggle file name (cubic_zirconia.csv).
"""
from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path

import pandas as pd

from ..paths import DATA_RAW
from ..utils import get_logger, load_json, save_json
from .registry import DatasetSpec, get_spec

log = get_logger("tabgen.data")


class ManualDownloadRequired(RuntimeError):
    pass


def raw_path(name: str) -> Path:
    return DATA_RAW / f"{name}.csv"


def meta_path(name: str) -> Path:
    return DATA_RAW / f"{name}.meta.json"


def is_available(name: str) -> bool:
    spec = get_spec(name)
    if raw_path(name).exists():
        return True
    manual = spec.source.get("file")
    return bool(manual and (DATA_RAW / manual).exists())


def _save(name: str, df: pd.DataFrame, target: str, numerical=None, categorical=None) -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path(name), index=False)
    save_json({"target": target, "numerical": numerical, "categorical": categorical}, meta_path(name))
    log.info(f"[{name}] saved {df.shape} -> {raw_path(name)}")
    return raw_path(name)


def _fetch_sklearn(spec: DatasetSpec) -> Path:
    from sklearn import datasets as skd
    loader = getattr(skd, spec.source["loader"])
    kwargs = {"as_frame": True}
    if spec.source["loader"].startswith("fetch_"):
        kwargs["data_home"] = str(DATA_RAW / "_sklearn")
    bunch = loader(**kwargs)
    df = bunch.frame
    target = spec.target if spec.target != "auto" else df.columns[-1]
    return _save(spec.name, df, target, numerical=[c for c in df.columns if c != target], categorical=[])


def _fetch_openml(spec: DatasetSpec) -> Path:
    from sklearn.datasets import fetch_openml
    bunch = fetch_openml(data_id=spec.source["data_id"], as_frame=True, parser="auto",
                         data_home=str(DATA_RAW / "_openml"))
    df = bunch.frame
    target = spec.target if spec.target != "auto" else bunch.target_names[0]
    feats = [c for c in df.columns if c != target]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feats if c not in num]
    df = df[feats + [target]]
    return _save(spec.name, df, target, numerical=num, categorical=cat)


def _fetch_urls(spec: DatasetSpec, urls: list[str]) -> Path | None:
    for url in urls:
        try:
            log.info(f"[{spec.name}] downloading {url}")
            tmp = DATA_RAW / f".{spec.name}.download"
            DATA_RAW.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            df = pd.read_csv(tmp, encoding_errors="replace")
            df.columns = [c.lstrip("﻿").strip() for c in df.columns]
            tmp.unlink(missing_ok=True)
            return _save(spec.name, df, spec.target, spec.numerical, spec.categorical)
        except Exception as e:  # noqa: BLE001 - try the next mirror
            log.warning(f"[{spec.name}] mirror failed ({e})")
    return None


def _fetch_kaggle(spec: DatasetSpec) -> Path:
    src = spec.source
    manual_file = DATA_RAW / src["file"]
    if manual_file.exists():
        df = pd.read_csv(manual_file)
        return _save(spec.name, df, spec.target, spec.numerical, spec.categorical)
    if src.get("urls") and (p := _fetch_urls(spec, src["urls"])):
        return p
    kaggle = shutil.which("kaggle")
    has_creds = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if kaggle and has_creds:
        out = DATA_RAW / "_kaggle" / spec.name
        out.mkdir(parents=True, exist_ok=True)
        cmd = [kaggle, "datasets", "download", "-d", src["slug"], "-p", str(out), "--unzip"]
        log.info(f"[{spec.name}] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        hits = list(out.rglob(src["file"]))
        if hits:
            return _save(spec.name, pd.read_csv(hits[0]), spec.target, spec.numerical, spec.categorical)
    raise ManualDownloadRequired(
        f"[{spec.name}] cannot be downloaded automatically.\n"
        f"  1. Open https://www.kaggle.com/datasets/{src['slug']}\n"
        f"  2. Download '{src['file']}' and put it at {manual_file}\n"
        f"  3. Re-run: python -m tabgen data download --dataset {spec.name}\n"
        f"  (or `pip install kaggle` + ~/.kaggle/kaggle.json for automatic download; see docs/DATASETS.md)")


def fetch(name: str, force: bool = False) -> Path:
    spec = get_spec(name)
    if raw_path(name).exists() and meta_path(name).exists() and not force:
        return raw_path(name)
    kind = spec.source["type"]
    if kind == "sklearn":
        return _fetch_sklearn(spec)
    if kind == "openml":
        return _fetch_openml(spec)
    if kind == "url":
        p = _fetch_urls(spec, spec.source["urls"])
        if p is None:
            raise RuntimeError(f"[{name}] all mirrors failed; see docs/DATASETS.md for manual steps")
        return p
    if kind == "kaggle":
        return _fetch_kaggle(spec)
    raise ValueError(f"Unknown source type {kind}")


def read_raw(name: str) -> tuple[pd.DataFrame, dict]:
    """Returns (raw dataframe exactly as downloaded, meta dict). Downloads on first use."""
    fetch(name)
    return pd.read_csv(raw_path(name), low_memory=False), load_json(meta_path(name))
