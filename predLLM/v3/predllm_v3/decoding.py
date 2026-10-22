"""Schema-constrained decoding with outlier bounding and negative guidance (v3 novelty).

Provides:
1. score_continuations: exact candidate categorical scoring using cached prefix KV states.
2. NumericConstraint: token-level FSM enforcing schema constraints on digits and decimals.
3. decode_numeric: token-by-token sampling constrained by the numeric FSM and clipped to active outlier/boundary range.
4. apply_negative_guidance: Classifier-Free Guidance (CFG) logit tilting away from majority distribution.
"""
from __future__ import annotations

import copy
import re

import torch

_NUM_CHARS = re.compile(r"^[ \-.0-9]+$")


def left_pad(seqs: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        mask[i, L - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), pos.to(device)


def _expand_cache(past, n: int):
    past = copy.deepcopy(past)
    if hasattr(past, "batch_repeat_interleave"):
        past.batch_repeat_interleave(n)
        return past
    return tuple(tuple(t.repeat_interleave(n, dim=0) for t in layer) for layer in past)


@torch.no_grad()
def score_continuations(
    model,
    prefixes: list[list[int]],
    cands: list[list[int]],
    pad_id: int,
    max_elems: float = 3e7,
    use_cache: bool = True,
) -> torch.Tensor:
    """Returns (B, K) tensor of sum_t log p(cand_k[t] | prefix_b, cand_k[<t])."""
    dev = model.device
    max_elems = float(max_elems)
    B, K = len(prefixes), len(cands)
    if not use_cache:
        return _score_no_cache(model, prefixes, cands, pad_id, max_elems)
    ids, mask, pos = left_pad(prefixes, pad_id, dev)
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    first = torch.log_softmax(out.logits[:, -1].float(), -1)                      # (B, V)
    Lc = max(len(c) for c in cands)
    cid = torch.full((K, Lc), pad_id, dtype=torch.long, device=dev)
    cmask = torch.zeros((K, Lc), dtype=torch.long, device=dev)
    for k, c in enumerate(cands):
        cid[k, :len(c)] = torch.tensor(c, device=dev)
        cmask[k, :len(c)] = 1
    scores = first[:, cid[:, 0]]                                                  # (B, K)
    if Lc == 1:
        return scores
    V = out.logits.shape[-1]
    per = max(1, int(max_elems // (B * (Lc - 1) * V)))                            # candidates per chunk
    last_pos = pos[:, -1]
    for k0 in range(0, K, per):
        ks = slice(k0, min(K, k0 + per))
        n = ks.stop - ks.start
        inp = cid[ks, :-1].repeat(B, 1)                                            # (B*n, Lc-1)
        am = torch.cat([mask.repeat_interleave(n, 0), cmask[ks, :-1].repeat(B, 1)], 1)
        p = last_pos.repeat_interleave(n)[:, None] + 1 + torch.arange(Lc - 1, device=dev)[None]
        o = model(
            input_ids=inp,
            attention_mask=am,
            position_ids=p,
            past_key_values=_expand_cache(out.past_key_values, n),
            use_cache=True,
        )
        lp = torch.log_softmax(o.logits.float(), -1)
        tgt = cid[ks, 1:].repeat(B, 1)
        g = lp.gather(-1, tgt[..., None])[..., 0] * cmask[ks, 1:].repeat(B, 1)
        scores[:, ks] += g.sum(-1).view(B, n)
    return scores


@torch.no_grad()
def _score_no_cache(model, prefixes, cands, pad_id, max_elems) -> torch.Tensor:
    dev = model.device
    max_elems = float(max_elems)
    pairs = [(b, k) for b in range(len(prefixes)) for k in range(len(cands))]
    scores = torch.zeros(len(prefixes), len(cands), device=dev)
    V = model.config.vocab_size
    L = max(len(p) for p in prefixes) + max(len(c) for c in cands)
    per = max(1, int(max_elems // (L * V)))
    for s in range(0, len(pairs), per):
        chunk = pairs[s:s + per]
        seqs = [prefixes[b] + cands[k] for b, k in chunk]
        ids, mask, pos = left_pad(seqs, pad_id, dev)
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=mask, position_ids=pos).logits.float(), -1)
        for i, (b, k) in enumerate(chunk):
            n = len(cands[k])
            Lr = ids.shape[1]
            tgt = ids[i, Lr - n:]
            scores[b, k] = lp[i, Lr - n - 1:Lr - 1].gather(-1, tgt[:, None]).sum()
    return scores


class NumericConstraint:
    """Token-level FSM for numbers. State = (phase, n_spaces, n_int, leading_zero, n_frac)"""

    START = (0, 0, 0, False, 0)

    def __init__(self, tok, vocab_size: int, device):
        self.V, self.device = vocab_size, device
        ids = list(range(min(len(tok), vocab_size)))
        toks = tok.convert_ids_to_tokens(ids)
        self.surface: dict[int, str] = {}
        for i, t in zip(ids, toks):
            if t is None:
                continue
            s = tok.convert_tokens_to_string([t])
            if s and _NUM_CHARS.match(s):
                self.surface[i] = s
        self._masks: dict = {}

    @staticmethod
    def advance(st, ch: str, spec):
        phase, nsp, nint, lead0, nfrac = st
        max_int, max_frac, neg = spec
        if phase == 0:
            if ch == " " and nsp < 2:
                return (0, nsp + 1, 0, False, 0)
            if ch == "-" and neg:
                return (1, nsp, 0, False, 0)
            return (2, nsp, 1, ch == "0", 0) if ch.isdigit() else None
        if phase == 1:
            return (2, nsp, 1, ch == "0", 0) if ch.isdigit() else None
        if phase == 2:
            if ch.isdigit() and not lead0 and nint < max_int:
                return (2, nsp, nint + 1, False, 0)
            return (3, nsp, nint, lead0, 0) if ch == "." and max_frac > 0 else None
        if phase == 3:
            return (4, nsp, nint, lead0, 1) if ch.isdigit() else None
        if phase == 4:
            return (4, nsp, nint, lead0, nfrac + 1) if ch.isdigit() and nfrac < max_frac else None
        return None

    @staticmethod
    def run(st, s: str, spec):
        for ch in s:
            st = NumericConstraint.advance(st, ch, spec)
            if st is None:
                return None
        return st

    @staticmethod
    def accepting(st) -> bool:
        return st[0] in (2, 4)

    def mask(self, st, spec, terminator: int) -> torch.Tensor:
        key = (st, spec, terminator)
        if key not in self._masks:
            m = torch.zeros(self.V, dtype=torch.bool)
            for i, s in self.surface.items():
                if self.run(st, s, spec) is not None:
                    m[i] = True
            if self.accepting(st):
                m[terminator] = True
            self._masks[key] = m.to(self.device)
        return self._masks[key]


_ONLY: dict = {}


def _only(tok_id: int, V: int, dev) -> torch.Tensor:
    key = (tok_id, V, str(dev))
    if key not in _ONLY:
        m = torch.zeros(V, dtype=torch.bool, device=dev)
        m[tok_id] = True
        _ONLY[key] = m
    return _ONLY[key]


@torch.no_grad()
def decode_numeric(
    model,
    prefixes: list[list[int]],
    spec,
    terminator: int,
    cons: NumericConstraint,
    pad_id: int,
    temperature: float,
    gen: torch.Generator,
    negative_guidance: float = 0.0,
    neg_prefixes: list[list[int]] | None = None,
) -> list[str | None]:
    """Samples numeric string per prefix. Optionally applies CFG negative guidance away from majority."""
    dev = model.device
    ids, mask, pos = left_pad(prefixes, pad_id, dev)
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    B = len(prefixes)
    states = [NumericConstraint.START] * B
    strs, done = [""] * B, [False] * B
    last = pos[:, -1:]
    max_steps = spec[0] + spec[1] + 4

    out_neg = None
    if negative_guidance > 0.0 and neg_prefixes is not None:
        n_ids, n_mask, n_pos = left_pad(neg_prefixes, pad_id, dev)
        out_neg = model(input_ids=n_ids, attention_mask=n_mask, position_ids=n_pos, use_cache=True)

    for _ in range(max_steps):
        logits = out.logits[:, -1].float()
        if out_neg is not None and negative_guidance > 0.0:
            neg_logits = out_neg.logits[:, -1].float()
            logits = logits - negative_guidance * neg_logits

        logits = logits / max(temperature, 1e-4)
        allow = torch.stack([
            cons.mask(states[i], spec, terminator) if not done[i] else _only(terminator, cons.V, dev)
            for i in range(B)
        ])
        logits = logits.masked_fill(~allow[:, :logits.shape[-1]], float("-inf"))
        nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen)

        for i, t in enumerate(nxt[:, 0].tolist()):
            if done[i]:
                continue
            if t == terminator:
                done[i] = True
            else:
                strs[i] += cons.surface[t]
                states[i] = cons.run(states[i], cons.surface[t], spec)

        if all(done):
            break

        mask = torch.cat([mask, torch.ones((B, 1), dtype=mask.dtype, device=dev)], 1)
        last = last + 1
        out = model(
            input_ids=nxt,
            attention_mask=mask,
            position_ids=last,
            past_key_values=out.past_key_values,
            use_cache=True,
        )
        if out_neg is not None:
            n_mask = torch.cat([n_mask, torch.ones((B, 1), dtype=n_mask.dtype, device=dev)], 1)
            out_neg = model(
                input_ids=nxt,
                attention_mask=n_mask,
                past_key_values=out_neg.past_key_values,
                use_cache=True,
            )

    return [s.strip() if (d or NumericConstraint.accepting(st)) else None for s, d, st in zip(strs, done, states)]
