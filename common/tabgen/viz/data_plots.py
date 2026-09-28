"""
Visualise a dataset at each preprocessing stage:
raw     - as downloaded: missing values per column, dtypes, first features' distributions
clean   - after column selection / NaN drop: distributions, class balance
encoded - after encoding + MinMax (what v1's LLM and XGBoost see): distributions, correlation
v2text  - example serialized rows exactly as the v1 and v2 LLMs see them
"""

from typing import Any, Optional, List, Dict, Union

def _grid(df, cols, title, path, max_cols):
    pass

def plot_dataset(name, stages, out):
    pass
