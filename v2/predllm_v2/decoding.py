"""
Schema-constrained decoding primitives (v2 novelty N2) - LLM USED HERE.

score_continuations : exact log p(candidate + terminator | prefix) for every candidate value, in one
                      prefix pass + batched candidate passes that reuse the prefix KV cache.
decode_numeric      : token-by-token sampling where every step's logits are masked by a finite-state
                      machine over the grammar  ' '{0,2} '-'? digit+ ('.' digit{1,p})?  with at most
                      `max_int` integer digits, then the column terminator. Invalid numbers are
                      impossible by construction.
"""

from typing import Any, Optional, List, Dict, Union

# Placeholder for constant / export
_NUM_CHARS: Any = None

def left_pad(seqs, pad_id, device):
    pass

def _expand_cache(past, n):
    """Repeat every batch row of a KV cache n times (row-major: b0,b0,..,b1,b1,..)."""
    pass

def score_continuations(model, prefixes, cands, pad_id, max_elems, use_cache):
    """Returns (B, K) tensor of sum_t log p(cand_k[t] | prefix_b, cand_k[<t])."""
    pass

def _score_no_cache(model, prefixes, cands, pad_id, max_elems):
    """Reference implementation (full recomputation); used by tests to validate the cached path."""
    pass

class NumericConstraint:
    """Token-level FSM for numbers. State = (phase, n_spaces, n_int, leading_zero, n_frac)
phase 0: before number (spaces/sign)  1: after '-'  2: integer digits  3: after '.'  4: fraction digits.
Accepting states: phase 2 or 4. Allowed-token masks are cached per (state, spec)."""
    def __init__(self, tok, vocab_size, device):
        pass

    @staticmethod
    def advance(st, ch, spec):
        pass

    @staticmethod
    def run(st, s, spec):
        pass

    @staticmethod
    def accepting(st):
        pass

    def mask(self, st, spec, terminator):
        pass

def decode_numeric(model, prefixes, spec, terminator, cons, pad_id, temperature, gen):
    """Samples one constrained number per prefix. Returns the number strings (None if unfinished)."""
    pass

def _only(tok_id, V, dev):
    pass
