"""
Module: common.tabgen.__main__
Responsibility: CLI entry point when invoked via `python -m tabgen`.
"""
import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
