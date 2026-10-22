"""SIR-Gibbs semantics and the P8 anonymised view -- no torch needed (fake engine)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tiltgen.core.serialize import LINK
from tiltgen.methods.pipeline import anonymize
from tiltgen.methods.v2_corrected import gibbs


class FakeEngine:
    """Records every scoring context; prefers the candidate closest to 'target'."""

    def __init__(self):
        self.contexts: list[str] = []

    def score(self, contexts, suffixes, add_eos=False):
        self.contexts.extend(contexts)
        rng = np.random.default_rng(len(self.contexts))
        return [list(rng.normal(size=len(s))) for s in suffixes]

    def encode(self, texts, eos=True):
        return [list(range(len(t) // 3 + 1)) for t in texts]


def test_gibbs_conditions_on_all_other_features(splits):
    s = splits.schema
    eng = FakeEngine()
    rows = splits.minor_heldout[s.features].head(5).reset_index(drop=True)
    out, diag = gibbs.sir_gibbs(rows, eng, splits, np.random.default_rng(0), sweeps=1,
                                n_candidates=4, max_rows=None)
    assert len(eng.contexts) == len(rows) * len(s.features)
    for ctx in eng.contexts:
        assert ctx.endswith(LINK.rstrip())                      # "..., col is"
        scored = ctx.rsplit(", ", 1)[1].split(LINK.rstrip())[0].strip()
        present = [c for c in s.features if f", {c}{LINK}" in ctx]
        # every OTHER feature is in the context, the scored one only as the last stub
        assert sorted(present) == sorted(c for c in s.features if c != scored)
    assert set(diag["change_rate"]) == set(s.features)
    assert out.shape == rows.shape


def test_gibbs_candidates_are_per_row_not_a_shared_pool(splits):
    s = splits.schema
    eng = FakeEngine()
    rows = splits.minor_heldout[s.features].head(40).reset_index(drop=True)
    out, _ = gibbs.sir_gibbs(rows, eng, splits, np.random.default_rng(1), sweeps=1,
                             n_candidates=4, max_rows=None)
    # palt2: every refined continuous value came from ONE pool of 8 copied values
    assert out["income"].nunique() > 8


def test_order_sensitivity_scores_whole_rows(splits):
    eng = FakeEngine()
    r = gibbs.order_sensitivity(splits.minor_heldout, eng, splits, np.random.default_rng(0),
                                n_rows=5, n_orders=4)
    assert r["synthetic"]["n_rows"] == 5 and r["synthetic"]["std_nats_per_token_mean"] > 0
    lo, hi = r["synthetic"]["std_nats_per_token_ci95"]
    assert lo <= r["synthetic"]["std_nats_per_token_mean"] <= hi


class OrderFreeEngine(FakeEngine):
    """A coherent joint: the score of a row does not depend on its clause order."""

    def score(self, contexts, suffixes, add_eos=False):
        return [[-float(len(set(t.split(", ")))) - len(t) * 0.0 for t in s] for s in suffixes]


def test_order_sensitivity_has_a_real_row_reference_and_zero_for_a_coherent_joint(splits):
    r = gibbs.order_sensitivity(splits.minor_heldout, OrderFreeEngine(), splits,
                                np.random.default_rng(0), n_rows=6, n_orders=5)
    assert r["real_source"] == "real_heldout_minority"
    assert r["synthetic"]["std_nats_per_token_mean"] == 0.0
    assert r["real"]["std_nats_per_token_mean"] == 0.0
    assert np.isnan(r["ratio_synthetic_to_real"])


def test_anonymised_view_roundtrip(splits):
    view, inv = anonymize(splits)
    assert view.schema.target == "y" and view.schema.features[0] == "f1"
    back = view.train_minor.rename(columns=inv)
    pd.testing.assert_frame_equal(back[splits.schema.features + [splits.schema.target]],
                                  splits.train_minor[splits.schema.features
                                                     + [splits.schema.target]])
