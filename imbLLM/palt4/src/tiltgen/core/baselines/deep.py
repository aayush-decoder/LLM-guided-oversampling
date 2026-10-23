"""Deep generative baselines (all optional extras).

    ctgan, tvae   SDV (pip install -e ".[sdv]"). CTGAN is the paper's runner-up
                  (Table II avg 0.7724); TVAE is also in Table II. Fit on D_train
                  (label included), sample minority rows conditionally on the label.
    tabddpm, arf  synthcity (pip install -e ".[synthcity]"). TabDDPM (Kotelnikov et
                  al., ICML 2023) is the diffusion reference a reviewer expects;
                  ARF (adversarial random forests, Watson et al., AISTATS 2023) is a
                  fast, strong non-neural generator. Fit on D_train with the label
                  as an ordinary column, then REJECTION-sample: generate in rounds
                  and keep rows whose label is the minority, until |largest class|
                  rows or `max_rounds`. The shortfall is reported, never padded.
                  (docs/RESEARCH_REVIEW.md, G5)

Not implemented, and why: TabSyn and TabDiff have no pip-installable API (research
code only); TabEBM and EPIC depend on TabPFN / a hosted chat LLM. Add them via
their official repositories if a reviewer asks; see docs/RESEARCH_REVIEW.md.
"""

from __future__ import annotations

import pandas as pd

from ..data.protocol import Splits

DEFAULT_EPOCHS = 300   # SDV default; the paper does not state one


def _synth(kind: str, splits: Splits, seed: int, epochs: int) -> pd.DataFrame:
    try:
        from sdv.metadata import Metadata
        from sdv.sampling import Condition
        from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    except ImportError:
        raise RuntimeError('CTGAN/TVAE need SDV: pip install -e ".[sdv]"') from None
    import numpy as np
    import torch
    s = splits.schema
    D = splits.imbalanced_train[[*s.features, s.target]].copy()
    meta = Metadata.detect_from_dataframe(D)
    for c in s.categorical + [s.target]:
        meta.update_column(column_name=c, sdtype="categorical")
    for c in s.continuous:
        meta.update_column(column_name=c, sdtype="numerical")
    np.random.seed(seed)
    torch.manual_seed(seed)
    cls = CTGANSynthesizer if kind == "ctgan" else TVAESynthesizer
    model = cls(meta, epochs=epochs, verbose=False)
    model.fit(D)
    out = model.sample_from_conditions(
        [Condition(num_rows=splits.n_target, column_values={s.target: s.minority_label})],
        max_tries_per_batch=200)
    return out[s.features].reset_index(drop=True)


def ctgan(splits: Splits, seed: int = 0, epochs: int | None = None) -> pd.DataFrame:
    return _synth("ctgan", splits, seed, epochs or DEFAULT_EPOCHS)


def tvae(splits: Splits, seed: int = 0, epochs: int | None = None) -> pd.DataFrame:
    return _synth("tvae", splits, seed, epochs or DEFAULT_EPOCHS)


def _synthcity(kind: str, splits: Splits, seed: int, epochs: int | None,
               max_rounds: int = 20) -> pd.DataFrame:
    try:
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader
    except ImportError:
        raise RuntimeError('TabDDPM/ARF need synthcity: pip install -e ".[synthcity]"') from None
    s = splits.schema
    D = splits.imbalanced_train[[*s.features, s.target]].copy()
    D[s.target] = D[s.target].astype(str)
    kw: dict = {"random_state": seed}
    if kind == "ddpm" and epochs:
        kw["n_iter"] = int(epochs)
    plugin = Plugins().get(kind, **kw)
    plugin.fit(GenericDataLoader(D, target_column=s.target))
    share = max(float((D[s.target] == s.minority_label).mean()), 1e-3)
    need, got = splits.n_target, []
    for _ in range(max_rounds):
        k = int(min(200_000, max(1000, 1.5 * (need - sum(map(len, got))) / share)))
        out = plugin.generate(count=k).dataframe()
        got.append(out[out[s.target].astype(str) == s.minority_label])
        if sum(map(len, got)) >= need:
            break
    res = pd.concat(got, ignore_index=True).head(need) if got else D.head(0)
    return res[s.features].reset_index(drop=True)


def tabddpm(splits: Splits, seed: int = 0, epochs: int | None = None) -> pd.DataFrame:
    return _synthcity("ddpm", splits, seed, epochs)


def arf(splits: Splits, seed: int = 0, epochs: int | None = None) -> pd.DataFrame:
    return _synthcity("arf", splits, seed, None)


REGISTRY = {"ctgan": ctgan, "tvae": tvae, "tabddpm": tabddpm, "arf": arf}
BACKEND = {"ctgan": "sdv", "tvae": "sdv", "tabddpm": "synthcity", "arf": "synthcity"}


def available(name: str | None = None) -> bool:
    """Is the backend of `name` (or of any deep baseline) installed?"""
    import importlib.util
    mods = {BACKEND[name]} if name else set(BACKEND.values())
    return any(importlib.util.find_spec(m) is not None for m in mods)
