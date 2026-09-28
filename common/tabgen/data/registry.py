"""
Reads configs/datasets.yaml into DatasetSpec objects and resolves dataset groups.
"""

from typing import Any, Optional, List, Dict, Union

class DatasetSpec:
    pass

def load_registry():
    pass

def get_spec(name):
    pass

def resolve_datasets(names):
    """Expands 'all' | 'paper' | 'user' | 'classification' | 'regression' | 'a,b,c' into dataset names."""
    pass
