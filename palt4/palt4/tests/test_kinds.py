"""Discrete (integer) and ordinal columns: core/data/kinds.py and the v2 flags
`snap_discrete` / `ordinal_interpolation` (docs/what_corrected.md, section H)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tiltgen.core.data import kinds
from tiltgen.core.data.protocol import prepare_frame
from tiltgen.methods.v2_corrected import gibbs
from tiltgen.methods.v2_corrected import interpolate as i2
from tiltgen.methods.v2_corrected import method as v2m

EDU = ("school", "bachelor", "master")


@pytest.fixture
def osplits(frame, spec):
    """The synthetic table with `edu` declared ordinal (school < bachelor < master)."""
    return prepare_frame(frame, replace(spec, ordinal=(("edu", EDU),)), seed=0, q=0.2)


def test_snap_goes_to_the_nearest_observed_value_and_keeps_gaps():
    sup = np.array([0.0, 2.0, 4.0])
    assert kinds.snap([0.9, 1.0, 1.1, 3.7, 9.0, -5.0], sup).tolist() == [0, 0, 2, 4, 4, 0]


def test_rank_roundtrip_and_rounding():
    import pandas as pd
    r = kinds.to_rank(pd.Series(["b", "a", "c"]), ["a", "b", "c"])
    assert r.tolist() == [1, 0, 2]
    assert kinds.from_rank([0.4, 0.6, 1.5, 7.0, -3.0], ["a", "b", "c"]).tolist() == \
        ["a", "b", "c", "c", "a"]


def test_protocol_detects_integer_columns_and_declared_ordinals(osplits, splits):
    assert "income" in osplits.schema.discrete           # rounded to whole numbers
    assert "age" not in osplits.schema.discrete          # one decimal
    assert osplits.schema.ordinal == {"edu": list(EDU)}
    assert splits.schema.ordinal == {}                   # nothing declared -> nothing assumed
    assert osplits.log["ordinal_skipped"] == {}


def test_an_order_that_misses_a_value_is_skipped_not_forced(frame, spec):
    s = prepare_frame(frame, replace(spec, ordinal=(("edu", ("school", "master")),)), seed=0)
    assert s.schema.ordinal == {} and "edu" in s.log["ordinal_skipped"]


def test_ordinal_interpolation_uses_ranks_and_stays_on_levels(osplits):
    s = osplits.schema
    inter, anchors, partners = i2.interpolate(osplits.train_minor, s, 400,
                                              np.random.default_rng(0), mode="knn",
                                              ordinal=s.ordinal)
    assert set(inter["edu"]) <= set(EDU)
    r_i = kinds.to_rank(osplits.train_minor["edu"], list(EDU))[anchors]
    r_j = kinds.to_rank(osplits.train_minor["edu"], list(EDU))[partners]
    r = kinds.to_rank(inter["edu"], list(EDU))
    # the new level always lies between the anchor's and the partner's level
    assert np.all((r >= np.minimum(r_i, r_j)) & (r <= np.maximum(r_i, r_j)))
    # and interpolation actually produces levels the anchor did not have
    assert (r != r_i).any()


def test_without_the_flag_ordinals_are_untouched(osplits):
    s = osplits.schema
    inter, _, _ = i2.interpolate(osplits.train_minor, s, 50, np.random.default_rng(0),
                                 mode="knn")
    assert "edu" not in inter.columns                    # v2 then copies the anchor's edu


def test_v2_corpus_snaps_integer_columns(osplits):
    cfg = v2m.V2Config(preset="recommended")
    texts, diag = v2m.build_corpus(osplits, np.random.default_rng(0), cfg=cfg)
    assert diag["interpolation"]["discrete_cells_snapped"] > 0
    assert diag["interpolation"]["ordinal_interpolated"] == ["edu"]
    incomes = [t.split("income is ")[1].split(",")[0] for t in texts if "income is " in t]
    assert incomes and all("." not in v for v in incomes)


def test_v1_exact_is_unchanged_by_the_new_flags(osplits):
    ex = v2m.V2Config(**v2m.PRESETS["v1_exact"])
    assert not ex.snap_discrete and not ex.ordinal_interpolation
    from tiltgen.methods.v1_paper import method as v1m
    a, _ = v2m.build_corpus(osplits, np.random.default_rng(3), cfg=ex)
    b, _ = v1m.build_corpus(osplits, np.random.default_rng(3), cfg=ex)
    assert a == b


class Scorer:
    def score(self, contexts, suffixes, add_eos=False):
        rng = np.random.default_rng(len(contexts))
        return [list(rng.normal(size=len(x))) for x in suffixes]

    def encode(self, texts, eos=True):
        return [[0] * (len(t) // 3 + 1) for t in texts]


def test_gibbs_keeps_integer_columns_on_their_support(osplits):
    s = osplits.schema
    rows = osplits.minor_heldout[s.features].head(12).reset_index(drop=True).copy()
    rows["income"] = rows["income"] + 0.37                 # an LLM wrote fractional counts
    out, diag = gibbs.sir_gibbs(rows, Scorer(), osplits, np.random.default_rng(0), sweeps=1,
                                n_candidates=6, max_rows=None, snap_discrete=True)
    sup = set(kinds.support(osplits.imbalanced_train, "income"))
    assert set(out["income"].astype(float)) <= sup
