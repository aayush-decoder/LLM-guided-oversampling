"""
Textual encoding (Sec. III-B1a), permutation (III-B1b) and parsing of generated text.

row [X1=v1, ..., XM=vM, Y=y]  ->  "X1 is v1, ..., XM is vM, target is y"
"""

from typing import Any, Optional, List, Dict, Union

def row_fields(row, features, task, decimals, target_decimals):
    """Formats every value of one encoded row as text (features rounded like the paper)."""
    pass

def column_order(features, strategy, rng, original, with_target):
    """permute_x: shuffle features, target last. permute_xy: shuffle everything. original: no shuffle."""
    pass

def encode(fields, order):
    pass

def parse(text, columns):
    """Inverse of `encode` for generated text. Keeps the first numeric value per known column and
ignores everything else (unknown names, repeated columns, non-numeric values)."""
    pass
