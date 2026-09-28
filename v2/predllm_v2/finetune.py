"""
v2 fine-tuning data (N1 + N3). Same D'_real composition as the paper (one permute_x copy, target
last, re-shuffled on every fetch, plus one original-order copy), but serialized in the schema's
canonical piece-wise form and returned with per-token loss weights for tabgen's weighted loop.
"""

from typing import Any, Optional, List, Dict, Union

class SchemaRowDataset(Dataset):
    def __init__(self, rows, codec, lam, w_scaffold, perm_copies, orig_copies, seed):
        pass

    def __len__(self):
        pass

    def order(self, i):
        pass

    def __getitem__(self, i):
        pass

    def text(self, i):
        pass
