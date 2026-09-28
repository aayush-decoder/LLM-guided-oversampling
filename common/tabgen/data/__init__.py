"""
Module: common.tabgen.data
Responsibility: Data registry, downloading, cleaning, and preprocessing utilities.
"""
from typing import Any

def fetch(*args, **kwargs):
    """Fetch a dataset from its configured source."""
    pass

def is_available(*args, **kwargs):
    """Check if dataset raw files are locally available."""
    pass

class ManualDownloadRequired(Exception):
    """Raised when a dataset requires manual download."""
    pass

__all__ = ["fetch", "is_available", "ManualDownloadRequired"]
