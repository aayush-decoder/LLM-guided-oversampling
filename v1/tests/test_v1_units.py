"""
Offline unit tests for v1 (no LLM download).
"""

from typing import Any, Optional, List, Dict, Union

# Placeholder for constant / export
FEATS: Any = None

def test_permute_x_keeps_target_last_and_permute_xy_moves_it():
    pass

def test_encode_parse_roundtrip():
    pass

def test_parse_rejects_garbage():
    pass

class CharTok:
    def __call__(self, text, **kw):
        pass

def test_dataset_is_permuted_plus_original():
    pass

def test_presets_match_table3():
    pass
