"""
Shared smoke harness: runs every given method once (seed 0) on a small dataset into
<version>/runs/_smoke/, then checks that each required artifact exists and passes the run audit.
Writes <version>/runs/_smoke/SMOKE_REPORT.md. Returns 0 when everything passes.
"""

from typing import Any, Optional, List, Dict, Union

def run_smoke(version, cfg, make_generator, methods, llm_tag, dataset):
    pass
