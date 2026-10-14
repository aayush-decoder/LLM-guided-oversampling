"""Field-wise twisted sequential Monte Carlo over an LLM row writer.

TARGET   q*(x) ∝ p_LM(x | prompt) · exp(λ z(x)) · 1[x valid]
         (p_LM is the fine-tuned generator, i.e. the proposal we can sample from)

A row is written field by field ("age is 41", then "job is admin", ...). With N
particles per group, at every field step t each unfinished particle extends its own
text by ONE field sampled from the LLM itself, and its log-weight is updated by
the twisted-SMC increment (Zhao et al., ICML 2024, Eq. 10, with proposal = model):

    log w_t  +=  log ψ_t(x_{1:t}) - log ψ_{t-1}(x_{1:t-1})
    log ψ_t   =  λ · ẑ(x_S)                   (partial-row twist, methods/tilt/partial.py)
    log ψ_T   =  λ · z(x) + log 1[x valid]    (exact terminal potential)

The increments telescope to λ z(x) + log 1[valid], so the weighted particles target
q* exactly whatever ẑ is (Prop. 3a in docs/THEORY.md); ẑ only decides WHEN weight
enters, i.e. how early bad partial rows are pruned by resampling. A particle whose
text stops parsing (unknown column, repeated column, bad number, unknown
category, EOS before all features) gets log ψ = -inf and dies at the next resample.

RESAMPLING  multinomial, within each group, when ESS < ess_frac · N.
OUTPUT      per group, `keep` finished particles drawn without replacement ∝ their
            final weights (Gumbel-top-k); keep <= N. keep = 1 is an exact draw from
            the particle approximation of q*. With `dedupe` (default) identical rows
            within a group count once: this avoids emitting copies but slightly
            flattens q* when rows repeat often (all-categorical tables).

The engine only needs `complete(prompts, ...) -> [(text, ended_with_eos)]`: the
field boundary is found by truncating each continuation at the first ", " (the
serialisation separator), so no custom decoding loop is required. Cost per output
row ≈ (N / keep) × (M + 1) short completions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...core.data.protocol import Schema
from ...core.serialize import _NUM, LINK, SEP, _repair
from .tilt import ess, normalised, select


def parse_clauses(body: str, schema: Schema, vocab: dict, repair: bool) -> tuple[dict, bool]:
    """Complete clauses 'col is v' of a partial row -> ({col: value}, ok)."""
    out: dict = {}
    for part in [p for p in body.split(SEP) if p.strip()]:
        col, sep, val = part.partition(LINK)
        col, val = col.strip(), val.strip()
        if not sep or col not in schema.features or col in out:
            if col == schema.target and sep:
                continue
            return out, False
        if col in schema.continuous:
            if not _NUM.match(val):
                return out, False
            out[col] = float(val)
        else:
            allowed = vocab.get(col, [])
            if val not in allowed:
                fixed = _repair(val, allowed) if repair else None
                if fixed is None:
                    return out, False
                val = fixed
            out[col] = val
    return out, True


@dataclass
class SMCStats:
    groups: int = 0
    particles: int = 0
    steps: int = 0
    resamples: int = 0
    completions: int = 0
    died_invalid: int = 0
    finished_valid: int = 0
    mean_final_ess: float = 0.0
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["resamples_per_group"] = round(self.resamples / max(self.groups, 1), 3)
        return d


def twisted_smc(engine, prompts: list[str], schema: Schema, *, vocab: dict, repair: bool,
                twist, terminal, n_particles: int = 8, keep: int = 4, ess_frac: float = 0.5,
                temperature: float = 0.7, top_p: float = 1.0, field_tokens: int = 24,
                batch_size: int = 64, rng: np.random.Generator, seed: int = 0,
                dedupe: bool = True) -> tuple[pd.DataFrame, SMCStats]:
    """twist(list[dict]) -> λ·ẑ for partial rows;  terminal(DataFrame) -> λ·z(x) with
    -inf for invalid rows (the caller applies clipping / snapping inside)."""
    M = len(schema.features)
    G, N = len(prompts), n_particles
    keep = min(keep, N)
    st = SMCStats(groups=G, particles=G * N)
    pre = [p for p in prompts for _ in range(N)]                 # full text so far
    grp = np.repeat(np.arange(G), N)
    logw = np.zeros(G * N)
    logpsi = np.zeros(G * N)
    done = np.zeros(G * N, bool)
    rows: list[dict | None] = [None] * (G * N)

    for t in range(2 * (M + 1)):
        act = np.flatnonzero(~done)
        if len(act) == 0:
            break
        st.steps += 1
        outs = engine.complete([pre[i] for i in act], temperature=temperature, top_p=top_p,
                               max_new_tokens=field_tokens, batch_size=batch_size,
                               seed=seed * 100_003 + t)
        st.completions += len(act)
        partial_idx, partial_rows = [], []
        full_idx, full_rows = [], []
        for i, (text, eos) in zip(act, outs):
            cut = text.find(SEP)
            if cut >= 0:                                          # one more complete field
                pre[i] += text[:cut] + ","
                finished = False
            else:
                pre[i] += text
                finished = bool(eos)
            # parse the WHOLE text: the prompt's own clauses ("Y is y_min, X_i is v") count
            # as written fields; a trailing clause cut mid-way (no separator yet) is left
            # out until it is complete
            body = pre[i].rstrip(",")
            if not (finished or pre[i].endswith(",")):
                body = SEP.join(body.split(SEP)[:-1])
            clauses, ok = parse_clauses(body, schema, vocab, repair)
            if not ok:
                logpsi_new = -np.inf
                done[i] = True
                st.died_invalid += 1
            elif len(clauses) == M:
                done[i] = True
                full_idx.append(i)
                full_rows.append(clauses)
                continue
            elif finished:
                logpsi_new = -np.inf                              # EOS before every feature
                done[i] = True
                st.died_invalid += 1
            else:
                partial_idx.append(i)
                partial_rows.append(clauses)
                continue
            logw[i] += logpsi_new - logpsi[i]
            logpsi[i] = logpsi_new
        if partial_idx:
            lp = np.asarray(twist(partial_rows), float)
            for i, v in zip(partial_idx, lp):
                logw[i] += v - logpsi[i]
                logpsi[i] = v
        if full_idx:
            lp = np.asarray(terminal(pd.DataFrame(full_rows)[schema.features]), float)
            for i, r, v in zip(full_idx, full_rows, lp):
                rows[i] = r
                logw[i] += v - logpsi[i]
                logpsi[i] = v
                st.finished_valid += int(np.isfinite(v))
        # resample within each group that still has unfinished particles
        for g in range(G):
            ii = np.flatnonzero(grp == g)
            if done[ii].all():
                continue
            w = normalised(logw[ii])
            if ess(w) < ess_frac * N and np.isfinite(logw[ii]).any():
                pick = ii[rng.choice(N, size=N, p=w)]
                snap = [(pre[j], logpsi[j], done[j], rows[j]) for j in pick]
                for j, (p_, lp_, d_, r_) in zip(ii, snap):
                    pre[j], logpsi[j], done[j], rows[j] = p_, lp_, d_, r_
                fin = np.isfinite(logw[ii])
                logw[ii] = np.log(np.sum(np.exp(logw[ii][fin] - logw[ii][fin].max()))) \
                    + logw[ii][fin].max() - np.log(N)
                st.resamples += 1

    out, ess_final = [], []
    for g in range(G):
        ii = np.array([i for i in np.flatnonzero(grp == g)
                       if done[i] and rows[i] is not None and np.isfinite(logw[i])])
        if len(ii) == 0:
            continue
        ess_final.append(ess(normalised(logw[ii])))
        seen, taken = set(), 0
        for j in ii[select(logw[ii], len(ii), rng)]:
            key = tuple(rows[j][c] for c in schema.features)
            if dedupe and key in seen:
                continue
            seen.add(key)
            out.append(rows[j])
            taken += 1
            if taken >= keep:
                break
    st.mean_final_ess = float(np.mean(ess_final)) if ess_final else 0.0
    frame = pd.DataFrame(out, columns=schema.features) if out else \
        pd.DataFrame(columns=schema.features)
    return frame, st
