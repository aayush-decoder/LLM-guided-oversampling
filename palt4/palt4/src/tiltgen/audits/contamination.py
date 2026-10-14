"""Memorisation / contamination probes on a PRE-TRAINED backbone (no fine-tuning).

Why (docs/RESEARCH_REVIEW.md, G3). The semantic-prior 2x2 (`tilt run prior2x2`)
asks whether the pre-trained prior helps. Its answer is ambiguous unless we also
know whether the backbone has simply SEEN the dataset: a gain on adult/german
could be recall, not modelling. Bordt et al. (COLM 2024, "Elephants never
forget") ran these probes on GPT-3.5/4 only; nobody has reported them for the
GPT-2-class backbones that GReaT / Pred-LLM / ImbLLM actually fine-tune.

Four probes, adapted from Bordt et al. to base (non-chat) models, all greedy:
    header     prompt = the first k header names of the raw CSV;
               score  = characters of the true remaining header reproduced
    row        prompt = header + `context_rows` consecutive raw CSV rows;
               score  = exact-match rate of the NEXT row (trials)
    feature    prompt = one row as "col = v, ..." without its most unique column;
               score  = exact-match rate of that column's value, vs the
                        rate of always answering its modal value
    first_tok  prompt = header + consecutive rows;
               score  = match rate of the next row's first cell vs the rate of
                        always answering the modal first cell

A probe is evidence of memorisation only where it clearly beats its baseline
(`*_baseline`). The raw file is read as downloaded (before cleaning), which is
the form a web-crawled copy would have.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from ..core.data import fetch


def _engine(model: str, hardware: str = "auto"):
    from ..core.llm import hardware as hw
    from ..core.llm import zoo
    from ..core.llm.engine import LLMEngine
    spec = zoo.get(model)
    plan = hw.plan(spec, hw.get_hardware(hardware), mode="auto", effective_batch=1,
                   seq_len=1024)
    return LLMEngine(spec, plan, init="pretrained").load()


def _csv_lines(df: pd.DataFrame) -> list[str]:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")   # default os.linesep is \r\n on Windows
    return buf.getvalue().rstrip("\n").split("\n")


def _first_line(text: str) -> str:
    return text.lstrip("\n").split("\n", 1)[0].strip()


def probe(name: str, engine, *, trials: int = 25, context_rows: int = 10,
          header_k: int = 3, seed: int = 0, max_chars: int = 1800) -> dict:
    df = pd.read_csv(fetch.raw_path(name))
    lines = _csv_lines(df)
    header, rows = lines[0], lines[1:]
    rng = np.random.default_rng(seed)
    out: dict = {"dataset": name, "n_rows": len(rows)}

    # -- header
    cols = header.split(",")
    k = min(header_k, max(1, len(cols) - 1))
    prompt = ",".join(cols[:k]) + ","
    truth = ",".join(cols[k:])
    comp = engine.complete([prompt], greedy=True, max_new_tokens=64)[0][0]
    comp = _first_line(comp)
    n = 0
    while n < min(len(comp), len(truth)) and comp[n] == truth[n]:
        n += 1
    out["header_chars_matched"] = n
    out["header_chars_total"] = len(truth)
    out["header_completion"] = comp[:120]

    # -- row completion and first cell
    starts = rng.integers(0, max(1, len(rows) - context_rows - 1), size=trials)
    prompts, nexts = [], []
    for st in starts:
        ctx = "\n".join([header, *rows[st:st + context_rows]]) + "\n"
        prompts.append(ctx[-max_chars:])
        nexts.append(rows[st + context_rows] if st + context_rows < len(rows) else rows[-1])
    comps = [_first_line(c) for c, _ in engine.complete(
        prompts, greedy=True, max_new_tokens=int(max(map(len, nexts)) / 2) + 16,
        batch_size=4)]
    out["row_exact_match"] = float(np.mean([c == t for c, t in zip(comps, nexts)]))
    first = lambda s: s.split(",", 1)[0].strip()  # noqa: E731
    firsts = [first(r) for r in rows]
    modal = pd.Series(firsts).mode().iloc[0]
    out["first_cell_match"] = float(np.mean([first(c) == first(t)
                                             for c, t in zip(comps, nexts)]))
    out["first_cell_baseline"] = float(np.mean([first(t) == modal for t in nexts]))

    # -- feature completion on the most unique column
    uniq = {c: df[c].nunique() / max(len(df), 1) for c in df.columns}
    target_col = max(uniq, key=uniq.get)
    idx = rng.choice(len(df), size=min(trials, len(df)), replace=False)
    fprompts, ftruth = [], []
    for i in idx:
        r = df.iloc[int(i)]
        parts = [f"{c} = {r[c]}" for c in df.columns if c != target_col]
        fprompts.append(", ".join(parts) + f", {target_col} =")
        ftruth.append(str(r[target_col]).strip())
    fc = [c.strip().split(",", 1)[0].strip() for c, _ in engine.complete(
        fprompts, greedy=True, max_new_tokens=16, batch_size=8)]
    fmodal = str(df[target_col].mode().iloc[0]).strip()
    out["feature_column"] = target_col
    out["feature_exact_match"] = float(np.mean([a == b for a, b in zip(fc, ftruth)]))
    out["feature_baseline"] = float(np.mean([t == fmodal for t in ftruth]))
    out["memorisation_flag"] = bool(
        out["row_exact_match"] > 0.1
        or out["feature_exact_match"] > out["feature_baseline"] + 0.1
        or out["first_cell_match"] > out["first_cell_baseline"] + 0.2
        or n >= 20)
    return out


def run(datasets: list[str], model: str, *, trials: int = 25, hardware: str = "auto",
        seed: int = 0) -> list[dict]:
    engine = _engine(model, hardware)
    res = []
    try:
        for d in datasets:
            try:
                r = probe(d, engine, trials=trials, seed=seed)
            except FileNotFoundError:
                r = {"dataset": d, "error": "not downloaded (tilt data fetch)"}
            r["model"] = model
            res.append(r)
    finally:
        engine.release()
    return res
