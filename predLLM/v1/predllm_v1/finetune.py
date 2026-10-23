"""Fine-tuning phase (Alg. 1 lines 2-12).

D'_real = { permute(s_i) } U { s_i }   (the second set only when augment_original=True)
The permutation is re-drawn every time an item is fetched, so each epoch sees fresh orders
(same as the authors' PredLLMDataset). Loss = plain autoregressive NLL (Eq. 1) via tabgen's loop.
"""
from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from .serialize import column_order, encode


class RowTextDataset(Dataset):
    def __init__(self, fields: list[dict], features: list[str], tok, permutation: str,
                 augment_original: bool, seed: int = 0):
        self.fields, self.features, self.tok = fields, features, tok
        self.permutation, self.augment = permutation, augment_original
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.fields) * (2 if self.augment else 1)

    def text(self, i: int) -> str:
        n = len(self.fields)
        order = column_order(self.features, self.permutation, self.rng, original=i >= n)
        return encode(self.fields[i % n], order)

    def __getitem__(self, i):
        ids = self.tok(self.text(i))["input_ids"] + [self.tok.eos_token_id]
        return {"input_ids": ids}
