"""TILT (methods/tilt): utilities, budget, selection, privacy gate, twist, SMC, the
SIR oversampler end to end with a fake generator, the membership attack, the
gain-vs-threshold metric, and the theory audits. No torch needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imbLLM.palt4.tests.conftest import needs_xgb

from tiltgen.audits import theory_checks
from tiltgen.core.evaluate import privacy as mia
from tiltgen.methods.tilt import method as tm
from tiltgen.methods.tilt import privacy, smc
from tiltgen.methods.tilt import tilt as T
from tiltgen.methods.tilt.partial import PartialUtility
from tiltgen.methods.tilt.utility import UTILITIES, UtilityModel


# ----------------------------------------------------------------- utility
@pytest.mark.parametrize("kind", [u for u in UTILITIES if u != "none"])
def test_every_utility_is_finite_and_varies(splits, kind):
    um = UtilityModel(kind=kind, seed=0).fit(splits)
    rows = pd.concat([splits.minor_heldout, splits.train_major.head(200)])[splits.schema.features]
    u = um.utility(rows)
    assert np.isfinite(u).all() and u.std() > 0
    if kind in ("fisher", "epistemic", "aleatoric"):
        assert (u >= -1e-12).all()


def test_validity_floor_keeps_minority_and_rejects_majority(splits):
    um = UtilityModel(kind="fisher", valid_q=0.05, seed=0).fit(splits)
    s = splits.schema
    assert um.valid(splits.minor_heldout[s.features]).mean() > 0.8
    assert um.valid(splits.train_major[s.features]).mean() < 0.3


def test_fisher_utility_is_higher_near_the_boundary(splits):
    """σ(1-σ) leverage: rows the probe is unsure about score higher than confident ones."""
    um = UtilityModel(kind="fisher", seed=0).fit(splits)
    rows = pd.concat([splits.minor_heldout, splits.train_major])[splits.schema.features]
    u, p = um.utility(rows), um.p_probe(rows)            # the probe that defines u
    # relative, not absolute: on this bimodal table a LINEAR probe never exceeds p = 0.2
    conf = np.abs(p - 0.5)
    unsure, sure = conf <= np.quantile(conf, 0.25), conf >= np.quantile(conf, 0.75)
    assert u[unsure].mean() > u[sure].mean()


# ----------------------------------------------------------------- budget / selection
def test_lambda_hits_the_kl_budget_and_zero_budget_is_no_tilt():
    z, _ = T.standardise(np.random.default_rng(0).normal(size=3000))
    lam, info = T.calibrate_lambda(z, 0.2)
    assert abs(info["kl"] - 0.2) < 5e-3 and info["binding"] == "kl_budget"
    assert T.calibrate_lambda(z, 0.0)[0] == 0.0


def test_selection_is_without_replacement_and_follows_the_weights():
    rng = np.random.default_rng(0)
    z = np.r_[np.zeros(500), np.full(500, 2.0)]
    idx = T.select(1.5 * z, 200, rng)
    assert len(set(idx)) == 200
    assert (idx >= 500).mean() > 0.8


# ----------------------------------------------------------------- privacy
def test_dcr_gate_drops_copies_and_keeps_fresh_rows(splits):
    s = splits.schema
    copies = splits.train_minor[s.features]
    keep_c, info = privacy.dcr_gate(copies, splits)
    keep_f, _ = privacy.dcr_gate(splits.minor_heldout[s.features], splits)
    assert keep_c.mean() < 0.05 and keep_f.mean() > 0.7 and info["tau"] > 0


def test_membership_attack_sees_copies_not_fresh_rows(splits):
    s = splits.schema
    leak = mia.mia_report(pd.concat([splits.train_minor] * 3)[s.features], splits)
    fresh = mia.mia_report(splits.train_major.head(300)[s.features], splits)
    assert leak["mia_auc"] > 0.75
    assert abs(fresh["mia_auc"] - 0.5) < abs(leak["mia_auc"] - 0.5)


# ----------------------------------------------------------------- twist and SMC
def test_partial_utility_predicts_full_row_utility(splits):
    s = splits.schema
    rows = pd.concat([splits.minor_heldout, splits.train_major.head(300)])[s.features]
    z, _ = T.standardise(UtilityModel(kind="fisher").fit(splits).utility(rows))
    pu = PartialUtility(s, seed=0).fit(rows, z)
    full = pu.predict(rows.to_dict("records"))
    assert np.corrcoef(full, z)[0, 1] > 0.7
    empty = pu.predict([{}] * 3)
    assert np.allclose(empty, empty[0])                 # no evidence -> one constant guess


def test_parse_clauses_rejects_bad_fields(splits):
    s = splits.schema
    vocab = {c: sorted(splits.train_minor[c].astype(str).unique()) for c in s.categorical}
    ok = smc.parse_clauses("target is yes, age is 30, edu is master", s, vocab, False)
    assert ok[1] and ok[0]["age"] == 30.0 and "target" not in ok[0]
    for bad in ("age is abc", "salary is 3", "age is 3, age is 4", "edu is phd"):
        assert not smc.parse_clauses(bad, s, vocab, False)[1]


def test_smc_is_exact_on_the_toy_row_model():
    name, ok, det = theory_checks.prop3_smc_exact(seed=0, groups=3000)
    assert det["flat_twist"]["tv_to_q*"] < 0.05


class RowWriter:
    """Fake LLM: writes the next missing feature of a random real minority row."""

    def __init__(self, rows: pd.DataFrame, schema, seed=1):
        self.rows, self.s = rows.reset_index(drop=True), schema
        self.rng = np.random.default_rng(seed)

    def complete(self, prompts, **kw):
        out = []
        for p in prompts:
            have = {part.split(" is ")[0].strip() for part in p.split(", ") if " is " in part}
            todo = [c for c in self.s.features if c not in have]
            r = self.rows.iloc[int(self.rng.integers(len(self.rows)))]
            if not todo:
                out.append(("", True))
                continue
            c = todo[0]
            v = r[c] if c in self.s.categorical else f"{float(r[c]):g}"
            last = len(todo) == 1
            out.append((f" {c} is {v}" + ("" if last else ", junk is x"), last))
        return out


def test_smc_writes_valid_rows_and_tilts_toward_high_utility(splits):
    s = splits.schema
    vocab = {c: sorted(splits.imbalanced_train[c].astype(str).unique()) for c in s.categorical}
    real = splits.minor_heldout
    def run(lam):
        term = lambda df: lam * (df["income"].astype(float).to_numpy() / 1e4)  # noqa: E731
        f, st = smc.twisted_smc(RowWriter(real, s), ["target is yes,"] * 120, s, vocab=vocab,
                                repair=False, twist=lambda r: np.zeros(len(r)), terminal=term,
                                n_particles=8, keep=2, rng=np.random.default_rng(0))
        return f, st
    base, st = run(0.0)
    tilted, _ = run(3.0)
    assert len(base) and set(base.columns) == set(s.features) and st.died_invalid == 0
    assert tilted["income"].mean() > base["income"].mean()


# ----------------------------------------------------------------- the oversampler
def _fake_pool(splits, n, seed=0):
    s = splits.schema
    rng = np.random.default_rng(seed)
    src = pd.concat([splits.minor_heldout, splits.train_major.sample(len(splits.minor_heldout),
                                                                     random_state=seed)])
    pool = src.sample(n=n, replace=True, random_state=seed)[s.features].reset_index(drop=True)
    pool["age"] = pool["age"] + rng.normal(0, 0.5, n)        # not exact copies
    return pool


@pytest.mark.parametrize("utility", ["fisher", "epistemic", "aleatoric", "none"])
def test_tilt_sir_selects_n_rows_within_budget(splits, monkeypatch, utility):
    from tiltgen.methods.pipeline import LLMOversampler
    monkeypatch.setattr(LLMOversampler, "generate",
                        lambda self, sp, n=None, seed=0: _fake_pool(sp, n, seed))
    cfg = tm.TiltConfig(**{**tm.PRESETS["tilt_sir"], "utility": utility, "kl_budget": 0.3,
                           "model": "tiny-random"})
    gen = tm.build(cfg)
    out = gen.generate(splits, n=100, seed=0)
    d = gen.diagnostics["tilt"]
    assert len(out) <= 100 and list(out.columns) == splits.schema.features
    assert d["kl"] <= 0.3 + 1e-3
    if utility == "none":
        assert d["lambda"] == 0.0
    else:
        assert d["z_selected_mean"] > d["z_pool_mean"]


def test_tilt_presets_resolve_and_contrast_needs_majority_rows():
    for name, p in tm.PRESETS.items():
        assert tm.TiltConfig(**p).sampler in tm.SAMPLERS, name
    with pytest.raises(ValueError):
        tm.TiltConfig(**{**tm.PRESETS["tilt_sir"], "utility": "contrast"})


# ----------------------------------------------------------------- report metric
def test_gain_vs_threshold_is_paired_per_dataset_and_seed():
    from tiltgen.experiments.report import add_gain_vs_threshold
    ok = pd.DataFrame({"dataset": ["a", "a", "a", "a"], "seed": [0, 0, 1, 1],
                       "method": ["threshold_only", "tilt_sir"] * 2,
                       "f1": [0.5, 0.4, 0.6, 0.3], "f1_tuned": [0.5, 0.55, 0.6, 0.58]})
    g = add_gain_vs_threshold(ok).set_index(["method", "seed"])["gain_vs_threshold"]
    assert g[("tilt_sir", 0)] == pytest.approx(0.05) and g[("tilt_sir", 1)] == pytest.approx(-0.02)


# ----------------------------------------------------------------- theory audits
@needs_xgb
def test_theory_audits_pass():
    for check in (theory_checks.lemma1_gibbs, theory_checks.prop2_first_order,
                  theory_checks.prop4_amplification, theory_checks.budget_monotone):
        name, ok, det = check(0)
        assert ok, (name, det)
