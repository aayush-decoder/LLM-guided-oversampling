"""TILT: Task-Informed LLM Tilting for imbalanced tabular oversampling.

    q*(x)  ∝  p_gen(x) · exp(λ z(x)) · 1[x label-valid] · 1[x passes the DCR gate]

p_gen is the fine-tuned generator of v2_corrected (preset `recommended`): the SAME
corpus and the SAME cached fine-tune, because every TILT knob acts at sampling
time. z is a standardised utility (methods/tilt/utility.py). λ is set by a KL
fidelity budget and capped by a record-amplification limit (methods/tilt/tilt.py).

SAMPLERS
    off   λ = 0: exactly v2 `recommended` (the ablation anchor)
    sir   whole rows: draw pool_factor · n rows from p_gen, weight by e^{λz}, keep n
          without replacement.  Cost ≈ pool_factor × plain generation.
    smc   field-wise twisted SMC (methods/tilt/smc.py) with a partial-row twist
          trained on a pilot pool.  Cost ≈ (particles / keep) × (M+1) field calls
          per row, but prunes bad partial rows early instead of wasting whole rows.

UTILITIES  fisher (default) | influence | epistemic | aleatoric (ablation) | none
           | contrast: u(x) = [log p_LM(x | y_maj) - log p_LM(x | y_min)] / tokens,
             the generator's own class log-ratio. A tilt by it IS sequence-level
             classifier-free guidance, p_min^{1-λ} p_maj^{λ} (docs/METHOD.md §4); it
             needs majority rows in the corpus (`majority_rows` > 0).

Everything is logged in diagnostics["tilt"]: λ, which constraint bound it, the
realised KL and ESS, validity / gate drop rates, and pool-vs-selected utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from ...core.data import kinds
from ...core.data.protocol import Splits
from ...core.serialize import category_vocab, clip_continuous, continuous_bounds, label_head
from ..pipeline import LLMOversampler, Strategies
from ..v1_paper.method import serialise_frames
from ..v2_corrected import method as v2m
from . import privacy, smc
from . import tilt as T
from .partial import PartialUtility
from .utility import UTILITIES, UtilityModel

SAMPLERS = ("off", "sir", "smc")
TILT_UTILITIES = UTILITIES + ("contrast",)


@dataclass
class TiltConfig(v2m.V2Config):
    sampler: str = "sir"
    utility: str = "fisher"
    kl_budget: float = 0.25            # nats: KL(q || p_gen) <= ε
    amp_cap: float = 20.0              # max e^{λ·range(z)} record amplification
    pool_factor: float = 4.0           # sir: pool = pool_factor · n
    pilot: int = 256                   # smc: rows used to calibrate λ and train the twist
    particles: int = 8
    keep: int = 4
    ess_frac: float = 0.5
    field_tokens: int = 24
    valid_q: float = 0.05              # validity floor = this quantile of real minority p_bal
    privacy_alpha: float = 0.05        # DCR gate quantile; 0 disables the gate
    probe_l2: float = 1e-2
    contrast_sign: float = 1.0         # +1: toward majority-like rows (boundary); -1: away
    majority_rows: float = 0.0         # majority rows added to the corpus, as a share of it

    def __post_init__(self):
        super().__post_init__()
        if self.sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}")
        if self.utility not in TILT_UTILITIES:
            raise ValueError(f"utility must be one of {TILT_UTILITIES}")
        if self.utility == "contrast" and self.majority_rows <= 0:
            raise ValueError("utility='contrast' needs majority rows in the corpus "
                             "(set majority_rows > 0, e.g. 1.0)")
        if not 1 <= self.keep <= self.particles:
            raise ValueError("need 1 <= keep <= particles")


# sampling-only presets on top of v2 `recommended` (one cached fine-tune serves all but
# `tilt_contrast`, whose corpus adds majority rows)
_BASE = {**v2m.PRESETS["recommended"], "measure_order_sensitivity": False,
         "measure_decomposition": False}
PRESETS = {
    "tilt_off": {**_BASE, "sampler": "off"},
    "tilt_sir": {**_BASE, "sampler": "sir", "utility": "fisher"},
    "tilt_smc": {**_BASE, "sampler": "smc", "utility": "fisher"},
    "tilt_influence": {**_BASE, "sampler": "sir", "utility": "influence"},
    "tilt_epistemic": {**_BASE, "sampler": "sir", "utility": "epistemic"},
    "tilt_aleatoric": {**_BASE, "sampler": "sir", "utility": "aleatoric"},
    "tilt_random": {**_BASE, "sampler": "sir", "utility": "none"},
    "tilt_contrast": {**_BASE, "sampler": "sir", "utility": "contrast", "majority_rows": 1.0},
    "tilt_nogate": {**_BASE, "sampler": "sir", "utility": "fisher", "valid_q": 0.0,
                    "privacy_alpha": 0.0},
}


def build_corpus(splits: Splits, rng: np.random.Generator, *, cfg: TiltConfig):
    texts, diag = v2m.build_corpus(splits, rng, cfg=cfg)
    if cfg.majority_rows > 0 and len(splits.train_major):
        k = int(min(len(splits.train_major), round(cfg.majority_rows * len(texts))))
        maj = splits.train_major.sample(n=k, random_state=int(rng.integers(1 << 30)))
        texts = texts + serialise_frames([maj], splits, cfg.permutation, rng, cfg.sig_digits)
        rng.shuffle(texts)
        diag["majority_rows_added"] = k
    return texts, diag


def strategies(cfg: TiltConfig) -> Strategies:
    base = v2m.strategies(cfg)
    base.corpus = partial(build_corpus, cfg=cfg)
    base.name = "tilt"
    return base


class TiltOversampler(LLMOversampler):
    """v2 generator + a tilted sampling stage. fit() is inherited unchanged."""

    # ------------------------------------------------------------------ helpers
    def _contrast(self, rows: pd.DataFrame, view: Splits) -> np.ndarray:
        """Per-token log-ratio log p(x | y_maj) - log p(x | y_min) under the fine-tuned LM."""
        from ...core.serialize import row_to_text
        s = view.schema
        order = list(s.features)
        texts = [" " + row_to_text(r, order, s, self.cfg.sig_digits)
                 for r in rows[s.features].to_dict("records")]
        h_min = label_head(s) + ","
        h_maj = label_head(s, s.majority_label) + ","
        lp_min = np.array([x[0] for x in self.engine.score([h_min] * len(texts),
                                                           [[t] for t in texts], add_eos=True)])
        lp_maj = np.array([x[0] for x in self.engine.score([h_maj] * len(texts),
                                                           [[t] for t in texts], add_eos=True)])
        ntok = np.array([len(x) for x in self.engine.encode(texts, eos=True)], float)
        return self.cfg.contrast_sign * (lp_maj - lp_min) / np.maximum(ntok, 1.0)

    def _to_view(self, rows: pd.DataFrame) -> pd.DataFrame:
        if not self._inverse:
            return rows
        fwd = {v: k for k, v in self._inverse.items()}
        return rows.rename(columns=fwd)

    def _to_orig(self, rows: pd.DataFrame) -> pd.DataFrame:
        return rows.rename(columns=self._inverse) if self._inverse else rows

    def _utility(self, rows_orig: pd.DataFrame, util: UtilityModel, view: Splits) -> np.ndarray:
        if self.cfg.utility == "contrast":
            return self._contrast(self._to_view(rows_orig), view)
        return util.utility(rows_orig)

    def _eligible(self, rows_orig: pd.DataFrame, util: UtilityModel, splits: Splits
                  ) -> tuple[np.ndarray, dict]:
        cfg = self.cfg
        valid = util.valid(rows_orig) if cfg.valid_q > 0 else np.ones(len(rows_orig), bool)
        if cfg.privacy_alpha > 0:
            gate, ginfo = privacy.dcr_gate(rows_orig, splits, alpha=cfg.privacy_alpha,
                                           seed=int(cfg.sig_digits))
        else:
            gate, ginfo = np.ones(len(rows_orig), bool), {"disabled": True}
        return valid & gate, {"valid_share": float(valid.mean()) if len(valid) else 0.0,
                              "tau_valid": util.tau_valid, "gate": ginfo}

    # ------------------------------------------------------------------ generate
    def generate(self, splits: Splits, n: int | None = None, seed: int = 0) -> pd.DataFrame:
        cfg = self.cfg
        if cfg.sampler == "off":
            return super().generate(splits, n=n, seed=seed)
        n = n or splits.n_target
        rng = np.random.default_rng(seed + 17)
        util = UtilityModel(kind="none" if cfg.utility == "contrast" else cfg.utility,
                            l2=cfg.probe_l2, valid_q=max(cfg.valid_q, 1e-9), seed=seed).fit(splits)
        if cfg.sampler == "sir":
            out, info = self._sir(splits, n, util, rng, seed)
        else:
            out, info = self._smc(splits, n, util, rng, seed)
        self.diagnostics["tilt"] = {"sampler": cfg.sampler, "utility": cfg.utility,
                                    "kl_budget": cfg.kl_budget, "amp_cap": cfg.amp_cap, **info}
        self.diagnostics["n_generated"] = int(len(out))
        self.diagnostics["shortfall"] = int(n - len(out))
        return out.reset_index(drop=True)

    def _sir(self, splits, n, util, rng, seed):
        cfg = self.cfg
        pool = super().generate(splits, n=int(np.ceil(cfg.pool_factor * n)), seed=seed)
        view = self._view(splits)
        if len(pool) == 0:
            return pool, {"pool": 0}
        ok, einfo = self._eligible(pool, util, splits)
        cand = pool[ok].reset_index(drop=True)
        if len(cand) == 0:
            return cand, {"pool": int(len(pool)), **einfo, "note": "no eligible rows"}
        u = self._utility(cand, util, view)
        z, zinfo = T.standardise(u)
        lam, cal = T.calibrate_lambda(z, cfg.kl_budget, amp_cap=cfg.amp_cap)
        idx = T.select(lam * z, n, rng)
        return cand.iloc[idx], {
            "pool": int(len(pool)), "eligible": int(len(cand)), **einfo, **cal,
            "utility_scale": zinfo, "pool_to_output": round(len(cand) / max(len(idx), 1), 3),
            "z_pool_mean": float(z.mean()), "z_selected_mean": float(z[idx].mean())}

    def _smc(self, splits, n, util, rng, seed):
        cfg = self.cfg
        view = self._view(splits)
        s = view.schema
        pilot = super().generate(splits, n=cfg.pilot, seed=seed)
        stats_pilot = self.parse_stats
        if len(pilot) < 10:
            return pilot, {"note": "pilot pool too small for SMC", "pilot": int(len(pilot))}
        ok, _ = self._eligible(pilot, util, splits)
        u_p = self._utility(pilot, util, view)
        mu, sd = float(np.mean(u_p[ok])) if ok.any() else 0.0, float(np.std(u_p))
        sd = sd if sd > 1e-12 else 1.0
        zf = lambda u: np.clip((np.asarray(u) - mu) / sd, -3, 3)  # noqa: E731
        lam, cal = T.calibrate_lambda(zf(u_p[ok]) if ok.any() else np.zeros(1), cfg.kl_budget,
                                      amp_cap=cfg.amp_cap)
        real = splits.train_minor[splits.schema.features]
        train_rows = pd.concat([pilot, real], ignore_index=True)
        train_z = np.concatenate([zf(u_p), zf(self._utility(real, util, view))])
        twist_model = PartialUtility(s, seed=seed).fit(self._to_view(train_rows), train_z)
        bounds = continuous_bounds(view.imbalanced_train, s)
        vocab = category_vocab(view.imbalanced_train, s)

        def post(frame_view: pd.DataFrame) -> pd.DataFrame:
            f = frame_view
            if cfg.clip_continuous:
                f, _ = clip_continuous(f, bounds)
            if cfg.snap_discrete and s.discrete:
                f, _ = kinds.snap_frame(f, s.discrete, view.imbalanced_train)
            return f

        def terminal(frame_view: pd.DataFrame) -> np.ndarray:
            orig = self._to_orig(post(frame_view))
            good, _ = self._eligible(orig, util, splits)
            v = lam * zf(self._utility(orig, util, view))
            return np.where(good, v, -np.inf)

        def twist(partials: list[dict]) -> np.ndarray:
            return lam * np.clip(twist_model.predict(partials), -3, 3)

        groups = int(np.ceil(n / cfg.keep))
        prompts, _ = self.strat.prompts(groups, view, rng)
        frame, st = smc.twisted_smc(
            self.engine, prompts, s, vocab=vocab, repair=cfg.repair_categories, twist=twist,
            terminal=terminal, n_particles=cfg.particles, keep=cfg.keep, ess_frac=cfg.ess_frac,
            temperature=cfg.temperature, top_p=cfg.top_p, field_tokens=cfg.field_tokens,
            batch_size=self.gen_batch(), rng=rng, seed=seed)
        self.parse_stats = stats_pilot
        out = self._to_orig(post(frame)).head(n)
        return out, {"pilot": int(len(pilot)), **cal, "smc": st.as_dict(),
                     "twist_training_rows": int(len(train_rows))}


def build(cfg: TiltConfig, **kw) -> TiltOversampler:
    return TiltOversampler(cfg, strategies(cfg), **kw)
