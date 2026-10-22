"""Config loading with ONE precedence rule, and stable config hashing.

PRECEDENCE (lowest -> highest), fixed and tested:

    dataclass defaults  <  preset  <  YAML file  <  command-line overrides

palt2 applied the preset AFTER the YAML, so `imb-v2 train --config full`
silently ran the `recommended` preset and dropped the YAML's own keys. Here a
YAML may name a preset (`preset: recommended`); the preset is expanded first
and every other key in the file overrides it; CLI flags override both.

The hash of the final resolved config names the run directory entry and the
model-cache key, so two runs that differ in any setting can never overwrite or
reuse each other's artefacts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import paths


def load_yaml(name_or_path: str | Path | None, folder: Path) -> dict:
    """Load `folder/<name>.yaml`, or an explicit path. None -> {}."""
    if not name_or_path:
        return {}
    p = Path(name_or_path)
    if not p.exists():
        p = folder / f"{name_or_path}.yaml"
    if not p.exists():
        avail = sorted(x.stem for x in folder.glob("*.yaml"))
        raise FileNotFoundError(f"No config {name_or_path!r}. Available in {folder}: {avail}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def resolve(cls, *, presets: Mapping[str, Mapping[str, Any]] | None = None,
            yaml_cfg: Mapping[str, Any] | None = None,
            cli: Mapping[str, Any] | None = None,
            preset: str | None = None):
    """Build a dataclass instance using the documented precedence."""
    yaml_cfg = dict(yaml_cfg or {})
    preset = preset or yaml_cfg.pop("preset", None)
    yaml_cfg.pop("preset", None)
    merged: dict[str, Any] = {}
    if preset:
        if not presets or preset not in presets:
            raise ValueError(f"Unknown preset {preset!r}. Have: {sorted(presets or {})}")
        merged.update(presets[preset])
    merged.update(yaml_cfg)
    merged.update({k: v for k, v in (cli or {}).items() if v is not None})
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(merged) - known
    if unknown:
        raise ValueError(f"Unknown config keys {sorted(unknown)} for {cls.__name__}. "
                         f"Known: {sorted(known)}")
    if preset and "preset" in known:
        merged["preset"] = preset
    return cls(**merged)


def as_dict(obj) -> dict:
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else dict(obj)


def stable_hash(obj: Any, n: int = 12) -> str:
    """sha1 of canonical JSON. Same config -> same hash, on every machine."""
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:n]


def experiment_config(name: str | None) -> dict:
    return load_yaml(name, paths.EXPERIMENT_CONFIGS)
