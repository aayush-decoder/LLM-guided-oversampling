"""
Schema + canonical piece-wise tokenization (v2 novelty N1, N3).

A row is serialized as a sequence of PIECES that are tokenized independently and concatenated:

    [BOS] key(c1) val(c1) SEP  key(c2) val(c2) SEP ... key(y) val(y) EOS
    key(c) = "c is" (first field) / " c is" (later fields),  val = " <value>",  SEP = ","

Because training (encode) and generation (sampler/posterior) build token ids from the SAME cached
pieces, the token sequence of a value never depends on its neighbours. That is what lets the decoder
(a) score every categorical value exactly and (b) write keys/separators itself.

Per-token loss weights (N3): key tokens -> w_scaffold (default 0, since the decoder writes them),
value tokens + their terminator (SEP/EOS) -> 1 for features and lambda for the target.
"""

from typing import Any, Optional, List, Dict, Union

class Schema:
    @property
    def columns(self):
        pass

    def fmt(self, col, v):
        pass

    def numeric_spec(self, col):
        pass

def _infer_decimals(s, cap):
    pass

def build_schema(split, serialization, max_decimals):
    """serialization='raw'     : real category strings and real units (v2 default, N1)
serialization='encoded' : the paper's MinMax/code space (ablation)"""
    pass

def rows_as_text(df, schema):
    pass

class FieldCodec:
    def __init__(self, schema, tok):
        pass

    def key_ids(self, col, first):
        pass

    def val_ids(self, s):
        pass

    def terminator(self, col):
        pass

    def encode(self, row, order, lam, w_scaffold):
        pass

    def prefix(self, row, order, then_key):
        """BOS + complete fields of `order` (each ended by SEP) [+ key of `then_key`]."""
        pass
