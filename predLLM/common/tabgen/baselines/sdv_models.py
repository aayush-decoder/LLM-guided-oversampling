"""Non-LLM baselines from the paper via SDV: CTGAN, TVAE, CopulaGAN.

They are trained in the *clean* space (real categories, real units), which is their native input and
the stronger setting, and their output is mapped to the encoded space by the shared preprocessor.
"""
from __future__ import annotations

import pandas as pd

SDV_METHODS = ("ctgan", "tvae", "copulagan")


def _metadata(df: pd.DataFrame, num_cols, cat_cols, target, task):
    try:
        from sdv.metadata import Metadata
        md = Metadata.detect_from_dataframe(df, table_name="t")
        upd = lambda c, t: md.update_column(column_name=c, sdtype=t, table_name="t")  # noqa: E731
    except ImportError:  # older SDV
        from sdv.metadata import SingleTableMetadata
        md = SingleTableMetadata()
        md.detect_from_dataframe(df)
        upd = lambda c, t: md.update_column(column_name=c, sdtype=t)  # noqa: E731
    for c in num_cols:
        upd(c, "numerical")
    for c in cat_cols:
        upd(c, "categorical")
    upd(target, "categorical" if task == "classification" else "numerical")
    return md


def sdv_fit_sample(method: str, train_clean: pd.DataFrame, num_cols, cat_cols, target, task, n: int,
                   epochs: int = 300, seed: int = 0, use_gpu: bool = True) -> pd.DataFrame:
    try:
        from sdv.single_table import CopulaGANSynthesizer, CTGANSynthesizer, TVAESynthesizer
    except ImportError as e:
        raise ImportError("SDV baselines need `pip install sdv` (see requirements.txt)") from e
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    cls = {"ctgan": CTGANSynthesizer, "tvae": TVAESynthesizer, "copulagan": CopulaGANSynthesizer}[method]
    md = _metadata(train_clean, num_cols, cat_cols, target, task)
    kw = {"epochs": epochs}
    gpu = use_gpu and torch.cuda.is_available()
    try:
        synth = cls(md, enable_gpu=gpu, **kw)
    except TypeError:
        synth = cls(md, cuda=gpu, **kw)
    synth.fit(train_clean)
    return synth.sample(num_rows=n)
