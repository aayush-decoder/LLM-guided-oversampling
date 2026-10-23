"""Reads configs/datasets.yaml into DatasetSpec objects and resolves dataset groups."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from ..paths import CONFIGS
from ..utils import load_yaml


@dataclass
class DatasetSpec:
    name: str
    task: str                                   # classification | regression
    source: dict
    target: str = "auto"
    numerical: list[str] | None = None
    categorical: list[str] | None = None
    drop: list[str] = field(default_factory=list)
    drop_prefix: list[str] = field(default_factory=list)
    dropna: bool = True
    binarize: str | None = None                 # None | "median"
    expected: dict = field(default_factory=dict)
    group: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def load_registry() -> dict[str, DatasetSpec]:
    raw = load_yaml(CONFIGS / "datasets.yaml")["datasets"]
    return {name: DatasetSpec(name=name, **cfg) for name, cfg in raw.items()}


def get_spec(name: str) -> DatasetSpec:
    reg = load_registry()
    if name not in reg:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(reg)}")
    return reg[name]


def resolve_datasets(names: str | list[str]) -> list[str]:
    """Expands 'all' | 'paper' | 'user' | 'classification' | 'regression' | 'a,b,c' into dataset names."""
    reg = load_registry()
    items = names.split(",") if isinstance(names, str) else list(names)
    out: list[str] = []
    for item in (i.strip() for i in items):
        if item == "all":
            out += list(reg)
        elif item in ("paper", "user"):
            out += [n for n, s in reg.items() if item in s.group]
        elif item in ("classification", "regression"):
            out += [n for n, s in reg.items() if s.task == item]
        else:
            get_spec(item)
            out.append(item)
    return list(dict.fromkeys(out))
