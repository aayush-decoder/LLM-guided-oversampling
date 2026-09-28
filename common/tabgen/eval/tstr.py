"""
Train-on-Synthetic / Test-on-Real (Fig. 1 of the paper): fit XGBoost (default hyper-parameters,
random_state=seed) on D_fake and score it on the real held-out D_test.
"""

from typing import Any, Optional, List, Dict, Union

def xgb_model(task, seed):
    pass

def tstr(train, test, features, task, n_classes, seed):
    pass

def primary_metric(task):
    """(metric name, higher_is_better)"""
    pass
