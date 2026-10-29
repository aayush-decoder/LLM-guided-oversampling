"""The paper's three LLM baselines, re-implemented on the SAME engine and pipeline.

These are "-style" re-implementations as ImbLLM Sec. II-B characterises them,
not the upstream packages -- so they share the backbone, schedule, parser and
evaluation with ImbLLM and differ only in the mechanism being compared:

    great_style    GReaT (Borisov et al. 2023): fine-tune on D_train with
                   permute_xy, sample with condition_y. No verification.
                   (= the `great_baseline` row of Table III)
    taptap_style   GReaT-style generation + an EXTERNAL XGBoost verifier trained
                   on the imbalanced D_train; rows it labels majority are
                   discarded (TapTap's verification, Zhang et al. 2023).
    predllm_style  GReaT-style generation + the fine-tuned LLM ITSELF re-queried
                   for the label ("<row>, Y is ?", scored over all classes, EOS-
                   terminated); rows it labels majority are discarded (Pred-LLM,
                   Nguyen et al. 2024).  [LLM CALL: engine.score]

The paper's Table V argument is exactly that these verifiers inherit the
imbalance of D_train; `rejected_by_verifier` in parse_stats.json measures it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..core.evaluate.downstream import Encoder
from ..core.serialize import SEP, LINK, clause
from .pipeline import LLMOversampler
from .v1_paper.method import V1Config, strategies as v1_strategies


def great_config(base: V1Config | None = None) -> V1Config:
    base = base or V1Config()
    return replace(base, condition="condition_y", permutation="permute_xy",
                   finetune_on="major+minor")


def _xgb_verifier(rows, splits, engine, rng):
    from xgboost import XGBClassifier
    s = splits.schema
    D = splits.imbalanced_train
    y = (D[s.target].astype(str) == s.minority_label).astype(int).to_numpy()
    enc = Encoder(s).fit(D)
    clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, verbosity=0,
                        tree_method="hist", random_state=0).fit(enc.transform(D), y)
    return clf.predict_proba(enc.transform(rows))[:, 1] >= 0.5


def _llm_verifier(rows, splits, engine, rng, sig: int = 5):
    s = splits.schema
    ctxs, cands = [], []
    for r in rows.to_dict("records"):
        order = list(rng.permutation(s.features))
        ctxs.append(SEP.join(clause(c, r[c], s, sig) for c in order) + SEP + s.target
                    + LINK.rstrip())
        cands.append([" " + c for c in s.classes])
    scores = np.array(engine.score(ctxs, cands, add_eos=True), float)
    return scores.argmax(1) == s.classes.index(s.minority_label)


def build(name: str, base: V1Config | None = None, **kw) -> LLMOversampler:
    cfg = great_config(base)
    st = v1_strategies(cfg)
    if name == "great_style":
        st.name = name
    elif name == "taptap_style":
        st.accept, st.name = _xgb_verifier, name
    elif name == "predllm_style":
        st.accept, st.name = _llm_verifier, name
    else:
        raise KeyError(f"unknown LLM baseline {name!r}; have {NAMES}")
    return LLMOversampler(cfg, st, **kw)


NAMES = ("great_style", "taptap_style", "predllm_style")
