"""
Downloads each dataset once into data/raw/<name>.csv, plus <name>.meta.json holding the target
and the column types.

Order of attempts: sklearn/openml/url auto-download -> public mirrors -> Kaggle API -> manual
(see docs/DATASETS.md). A manually placed file may use either the registry name (gem_price.csv)
or the original Kaggle file name (cubic_zirconia.csv).
"""

from typing import Any, Optional, List, Dict, Union

class ManualDownloadRequired(RuntimeError):
    pass

def raw_path(name):
    pass

def meta_path(name):
    pass

def is_available(name):
    pass

def _save(name, df, target, numerical, categorical):
    pass

def _fetch_sklearn(spec):
    pass

def _fetch_openml(spec):
    pass

def _fetch_urls(spec, urls):
    pass

def _fetch_kaggle(spec):
    pass

def fetch(name, force):
    pass

def read_raw(name):
    """Returns (raw dataframe exactly as downloaded, meta dict). Downloads on first use."""
    pass
