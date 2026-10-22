"""Row <-> sentence. Shared by every LLM-based method.

FORMAT (paper Fig. 3a)            "Income is >50K, Edu is Bachelor, WH is 40.5"
    clauses "<col> is <value>" joined by ", ". prepare() has already removed
    ',' and ' is ' from every categorical value and column name, so splitting
    is unambiguous (palt2 split on every comma inside values).

PROMPTS END WITH "," -- NOT ", "
    In training text a clause boundary is tokenised as ","  + " Edu" (the space
    belongs to the NEXT word under GPT-2 / Qwen / Llama-3 byte-level BPE). A
    prompt ending in ", " instead ends on a lone-space token the model never saw
    before a column name. palt2 did this; here prompts end with "," and the
    model emits " Edu ..." exactly as in training.

NUMBERS
    Formatted to 5 significant digits (not 2 decimals). palt2 rounded every
    number to 2 decimals, which collapses features like `mean smoothness`
    (~0.1 +- 0.014) onto a handful of values.

PARSE ACCOUNTING
    ParseStats.yield_rate = parsed / completions (rows actually sampled).
    palt2 divided by the number *requested*, so oversampled prompts made a 60%
    true yield look like 90% and yields could exceed 1.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .data.protocol import Schema

SEP = ", "
LINK = " is "
_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


# --------------------------------------------------------------------------- format

def fmt_num(x, sig: int = 5) -> str:
    f = float(x)
    if np.isfinite(f) and f == round(f) and abs(f) < 1e15:
        return str(int(round(f)))
    return np.format_float_positional(f, precision=sig, unique=True, fractional=False,
                                      trim="-")


def fmt_value(v, is_continuous: bool, sig: int = 5) -> str:
    return fmt_num(v, sig) if is_continuous else str(v)


def clause(col: str, v, schema: Schema, sig: int = 5) -> str:
    return f"{col}{LINK}{fmt_value(v, col in schema.continuous, sig)}"


def row_to_text(row, cols: list[str], schema: Schema, sig: int = 5) -> str:
    """Serialise `row` visiting `cols` in that order (the order IS the strategy).
    `cols` may include the target."""
    con = set(schema.continuous)
    return SEP.join(f"{c}{LINK}{fmt_value(row[c], c in con, sig)}" for c in cols)


def label_head(schema: Schema, label: str | None = None) -> str:
    return f"{schema.target}{LINK}{label if label is not None else schema.minority_label}"


def prompt_from_clauses(clauses: list[str]) -> str:
    """Prompt = clauses joined by ', ' and terminated by ',' (see module doc)."""
    return SEP.join(clauses) + ","


# --------------------------------------------------------------------------- parse

@dataclass
class ParseStats:
    prompts_sent: int = 0
    completions: int = 0
    finished_with_eos: int = 0
    parsed: int = 0
    dropped_missing_features: int = 0
    dropped_bad_number: int = 0
    dropped_unknown_category: int = 0
    repaired_category: int = 0
    unknown_column_clauses: int = 0
    rejected_by_verifier: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def yield_rate(self) -> float:
        return self.parsed / max(self.completions, 1)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["yield_rate"] = round(self.yield_rate, 4)
        d["eos_rate"] = round(self.finished_with_eos / max(self.completions, 1), 4)
        return d

    def _note(self, msg: str) -> None:
        if len(self.notes) < 8:
            self.notes.append(msg)


def _repair(raw: str, allowed: list[str]) -> str | None:
    """Case-insensitive exact match, else a UNIQUE prefix match of length >= 2.
    Empty or ambiguous values are rejected (palt2 snapped "" to the first
    category because every string starts with "")."""
    low = raw.strip().lower()
    if len(low) == 0:
        return None
    exact = [a for a in allowed if a.lower() == low]
    if len(exact) == 1:
        return exact[0]
    if len(low) < 2:
        return None
    hits = [a for a in allowed if a.lower().startswith(low) or low.startswith(a.lower())]
    return hits[0] if len(hits) == 1 else None


def text_to_row(text: str, schema: Schema, *, vocab: dict[str, list[str]] | None = None,
                repair: bool = False, stats: ParseStats | None = None,
                require_target: str | None = None) -> dict | None:
    """Parse one generated sentence -> {feature: value} or None.

    require_target: if given, a target clause must be present with this value
    (used by the label-query baselines); otherwise target clauses are ignored.
    """
    text = text.split("\n")[0].strip().rstrip(",").strip()
    found: dict[str, str] = {}
    for part in text.split(SEP):
        col, sep, val = part.partition(LINK)
        if not sep:
            continue
        col, val = col.strip(), val.strip()
        if col in found:
            continue
        if col in schema.features or col == schema.target:
            found[col] = val
        elif stats:
            stats.unknown_column_clauses += 1

    missing = [c for c in schema.features if c not in found]
    if missing:
        if stats:
            stats.dropped_missing_features += 1
            stats._note(f"missing {missing[:3]}: {text[:80]!r}")
        return None

    out: dict = {}
    for c in schema.continuous:
        v = found[c]
        if not _NUM.match(v):
            if stats:
                stats.dropped_bad_number += 1
                stats._note(f"bad number {c}={v!r}")
            return None
        out[c] = float(v)
    for c in schema.categorical:
        v = found[c]
        if vocab is not None and v not in vocab.get(c, ()):
            fixed = _repair(v, vocab.get(c, [])) if repair else None
            if fixed is None:
                if stats:
                    stats.dropped_unknown_category += 1
                    stats._note(f"unknown category {c}={v!r}")
                return None
            if stats:
                stats.repaired_category += 1
            v = fixed
        out[c] = v
    if require_target is not None and found.get(schema.target) != require_target:
        return None
    return out


def texts_to_frame(texts: list[str], schema: Schema, **kw) -> tuple[pd.DataFrame, list[int]]:
    """Parse many; returns (frame, indices of the texts that parsed)."""
    rows, keep = [], []
    stats = kw.get("stats")
    for i, t in enumerate(texts):
        r = text_to_row(t, schema, **kw)
        if r is not None:
            rows.append(r)
            keep.append(i)
    if stats:
        stats.parsed += len(rows)
    frame = pd.DataFrame(rows, columns=schema.features) if rows else \
        pd.DataFrame(columns=schema.features)
    return frame, keep


def category_vocab(df: pd.DataFrame, schema: Schema) -> dict[str, list[str]]:
    return {c: sorted(df[c].astype(str).unique()) for c in schema.categorical}


def continuous_bounds(df: pd.DataFrame, schema: Schema) -> dict[str, tuple[float, float]]:
    return {c: (float(df[c].min()), float(df[c].max())) for c in schema.continuous}


def clip_continuous(df: pd.DataFrame, bounds: dict) -> tuple[pd.DataFrame, int]:
    """Clamp into the observed training range. Returns (frame, #cells clipped)."""
    df = df.copy()
    n = 0
    for c, (lo, hi) in bounds.items():
        if c in df.columns and len(df):
            x = df[c].astype(float)
            n += int(((x < lo) | (x > hi)).sum())
            df[c] = x.clip(lo, hi)
    return df, n
