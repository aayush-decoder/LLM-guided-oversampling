"""
Three data stages, used by every generator and by the visualisations:

  raw      exactly as downloaded                                  (download.read_raw)
  clean    selected features, stripped strings, NaN rows dropped,
           categorical -> str, numerical -> float, target first    (clean_table)
  encoded  paper protocol (Pred-LLM read_data.py): categories -> integer codes, then MinMax
           scaling of every feature, rounded to 2 decimals; class labels -> 0..K-1    (TablePreprocessor)

TSTR evaluation always happens in the *encoded* space, so every method (v1, v2, SDV) is scored on
identical features.
"""

from typing import Any, Optional, List, Dict, Union

# Placeholder for constant / export
TARGET: Any = None

class CleanTable:
    pass

def _natural_sort(values):
    pass

def _label_str(v):
    pass

def clean_table(name, binarize):
    pass

class TablePreprocessor:
    """Paper-faithful encoder: categorical -> code, every feature MinMax-scaled to [0, 1] and rounded."""
    def __init__(self, num_cols, cat_cols, features, target, task, decimals):
        pass

    def fit(self, df):
        pass

    def _codes(self, df):
        pass

    def transform(self, df):
        """clean-space -> encoded-space. Rows whose class label is unknown are dropped."""
        pass

    def inverse_transform(self, enc):
        """encoded-space -> clean-space (codes rounded to the nearest category)."""
        pass

    def to_dict(self):
        pass

    @classmethod
    def from_dict(cls, d):
        pass

class TabularSplit:
    """One train/test split of one dataset for one seed, in both clean and encoded space."""
    @property
    def n_classes(self):
        pass

    def summary(self):
        pass

def make_split(name, seed, test_size, train_size, fit_scaler_on, binarize, max_rows, decimals):
    pass
