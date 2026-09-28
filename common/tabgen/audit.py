"""
Audits. Each writes a JSON report and prints PASS/WARN/FAIL lines.

env   python/torch/CUDA/VRAM, key packages, which registry LLMs fit this GPU
data  per dataset: availability, size vs registry expectation, missing values, duplicates,
      constant columns, class imbalance, train/test overlap, single-feature target leakage
run   per finished run dir: required artifacts, synthetic validity (NaN / range / class coverage),
      memorisation (exact copies of training rows), label-distribution shift, metric sanity
"""

from typing import Any, Optional, List, Dict, Union

# Placeholder for constant / export
REQUIRED_RUN_FILES: Any = None

class Report:
    def __init__(self, title):
        pass

    def add(self, level, msg):
        pass

    def ok(self, cond, msg_ok, msg_bad, bad):
        pass

    @property
    def status(self):
        pass

def audit_env():
    pass

def audit_data(name):
    pass

def audit_run(run_dir):
    pass
