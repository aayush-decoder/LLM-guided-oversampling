"""v2 fine-tuning data (N1 + N3). Same D'_real composition as the paper (one permute_x copy, target
last, re-shuffled on every fetch, plus one original-order copy), but serialized in the schema's
canonical piece-wise form and returned with per-token loss weights for tabgen's weighted loop."""
from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from .codec import FieldCodec


class SchemaRowDataset(Dataset):
    def __init__(self, rows: list[dict], codec: FieldCodec, lam: float, w_scaffold: float,
                 perm_copies: int = 1, orig_copies: int = 1, seed: int = 0):
        self.rows, self.codec, self.lam, self.w_scaffold = rows, codec, lam, w_scaffold
        self.perm_copies, self.orig_copies = perm_copies, orig_copies
        self.rng = np.random.default_rng(seed)
        self.feats = codec.schema.features

    def __len__(self):
        return len(self.rows) * (self.perm_copies + self.orig_copies)

    def order(self, i: int) -> list[str]:
        if i // len(self.rows) < self.perm_copies:
            f = [self.feats[j] for j in self.rng.permutation(len(self.feats))]
        else:
            f = list(self.feats)
        return f + [self.codec.schema.target]

    def __getitem__(self, i):
        ids, w = self.codec.encode(self.rows[i % len(self.rows)], self.order(i), self.lam, self.w_scaffold)
        return {"input_ids": ids, "weights": w}

    def text(self, i: int) -> str:
        return self.codec.tok.decode(self[i]["input_ids"])
