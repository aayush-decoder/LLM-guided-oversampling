"""Direct entrypoint to execute all versions (v1, v2, v3) across all datasets.

Equivalent to: python run_all_versions.py --datasets all

Usage:
  python run_all_datasets.py
  python run_all_datasets.py --preset quick
  python run_all_datasets.py --epochs 15
"""
import sys
from run_all_versions import main

if __name__ == "__main__":
    # Ensure default dataset is 'all' unless specified
    if not any(arg in sys.argv for arg in ("--dataset", "--datasets")):
        sys.argv.extend(["--datasets", "all"])
    sys.exit(main() or 0)
