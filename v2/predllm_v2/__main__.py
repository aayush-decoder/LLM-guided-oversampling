"""
Module: v2.predllm_v2.__main__
Responsibility: CLI entry point when invoked via `python -m predllm_v2`.
"""
import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
