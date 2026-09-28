"""
Small helpers: seeding, config loading/overrides, JSON io, logging, device selection.
"""

from typing import Any, Optional, List, Dict, Union

def set_seed(seed):
    pass

def load_yaml(path):
    pass

def deep_update(base, upd):
    pass

def apply_overrides(cfg, overrides):
    """Apply `a.b.c=value` overrides (value parsed as YAML, so 5 -> int, true -> bool)."""
    pass

def _json_default(o):
    pass

def save_json(obj, path):
    pass

def load_json(path):
    pass

def get_logger(name):
    pass

def pick_device(pref):
    pass

class Timer:
    def __enter__(self):
        pass

    def __exit__(self, *exc):
        pass

def format_number(v, decimals):
    """Canonical number -> text: rounded, no trailing zeros, no '-0' ("0.50" -> "0.5", "3.0" -> "3")."""
    pass
