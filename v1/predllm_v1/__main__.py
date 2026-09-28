"""
Module: v1.predllm_v1.__main__
Responsibility: CLI entry point when invoked via `python -m predllm_v1`.
"""
import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
