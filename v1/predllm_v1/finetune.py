"""
Fine-tuning phase (Alg. 1 lines 2-12).

D'_real = { permute(s_i) } U { s_i }   (the second set only when augment_original=True)
The permutation is re-drawn every time an item is fetched, so each epoch sees fresh orders
(same as the authors' PredLLMDataset). Loss = plain autoregressive NLL (Eq. 1) via tabgen's loop.
"""

from typing import Any, Optional, List, Dict, Union

class RowTextDataset(Dataset):
    def __init__(self, fields, features, tok, permutation, augment_original, seed):
        pass

    def __len__(self):
        pass

    def text(self, i):
        pass

    def __getitem__(self, i):
        pass
