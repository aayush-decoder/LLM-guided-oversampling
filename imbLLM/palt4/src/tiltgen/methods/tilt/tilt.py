"""The exponential tilt and its budget.

TARGET        q_λ(x) ∝ p_gen(x) · exp(λ z(x)),    z = standardised utility
WHY THIS FORM Gibbs variational principle (docs/THEORY.md, Lemma 1):
                  argmax_q  E_q[z] - (1/λ) KL(q || p)  =  p · exp(λ z) / Z
              so λ is the Lagrange multiplier of a KL fidelity budget
                  KL(q || p_gen) <= ε.
STANDARDISE   z = clip((u - mean) / sd, ±3) on the pilot pool, so λ is in
              "nats per standard deviation of utility" for every dataset and
              every utility; the budget ε then fixes λ anyway.
BUDGET        On a pool x_1..x_N ~ p_gen with self-normalised weights w̃_i ∝ e^{λ z_i}:
                  KL(q_λ || p_gen) ≈ Σ_i w̃_i log(N w̃_i)       (increasing in λ)
              λ_ε solves KL = ε by bisection.
PRIVACY CAP   the most a record's neighbourhood can be amplified is e^{λ·range(z)}
              (docs/THEORY.md, Prop. 4), so λ <= log(A) / range(z).
              Final λ = min(λ_ε, λ_A).
SELECTION     m of N pool rows WITHOUT replacement by Gumbel-top-k on log w
              (Efraimidis-Spirakis successive sampling): never duplicates a row.
              For m << N this approaches i.i.d. draws from q_λ; the report
              logs N/m and the ESS.
"""

from __future__ import annotations

import numpy as np


def standardise(u: np.ndarray, clip: float = 3.0) -> tuple[np.ndarray, dict]:
    u = np.asarray(u, float)
    mu, sd = float(np.mean(u)), float(np.std(u))
    z = np.zeros_like(u) if sd <= 1e-12 else np.clip((u - mu) / sd, -clip, clip)
    return z, {"mean": mu, "sd": sd, "clip": clip}


def normalised(logw: np.ndarray) -> np.ndarray:
    logw = np.asarray(logw, float)
    fin = np.isfinite(logw)
    if not fin.any():
        return np.full(len(logw), 1.0 / max(len(logw), 1))
    w = np.where(fin, np.exp(logw - logw[fin].max()), 0.0)
    return w / w.sum()


def kl_to_proposal(w: np.ndarray) -> float:
    """KL(q || p) for q = self-normalised weights on N proposal draws."""
    w = np.asarray(w, float)
    n = len(w)
    nz = w > 0
    return float(np.sum(w[nz] * np.log(n * w[nz])))


def ess(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    return float(1.0 / np.sum(w ** 2)) if w.sum() > 0 else 0.0


def calibrate_lambda(z: np.ndarray, epsilon: float, *, amp_cap: float | None = None,
                     lam_max: float = 50.0, tol: float = 1e-4) -> tuple[float, dict]:
    """Largest λ with KL(q_λ || p_gen) <= ε, capped by the amplification limit."""
    z = np.asarray(z, float)
    rng_z = float(z.max() - z.min()) if len(z) else 0.0
    lam_amp = np.inf if not amp_cap or rng_z <= 0 else float(np.log(amp_cap) / rng_z)
    if epsilon <= 0 or rng_z <= 0:
        lam_eps = 0.0
    else:
        lo, hi = 0.0, lam_max
        if kl_to_proposal(normalised(hi * z)) <= epsilon:
            lo = hi
        else:
            while hi - lo > tol:
                mid = 0.5 * (lo + hi)
                if kl_to_proposal(normalised(mid * z)) > epsilon:
                    hi = mid
                else:
                    lo = mid
        lam_eps = lo
    lam = float(min(lam_eps, lam_amp))
    w = normalised(lam * z)
    return lam, {"lambda": lam, "lambda_budget": lam_eps,
                 "lambda_amp_cap": None if not np.isfinite(lam_amp) else lam_amp,
                 "binding": "amplification" if lam_amp < lam_eps else "kl_budget",
                 "kl": kl_to_proposal(w), "ess": ess(w), "n_pool": int(len(z)),
                 "max_amplification": float(np.exp(lam * rng_z))}


def select(logw: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of m rows sampled without replacement ∝ exp(logw) (Gumbel-top-k)."""
    logw = np.asarray(logw, float)
    m = int(min(m, np.isfinite(logw).sum()))
    keys = np.where(np.isfinite(logw), logw + rng.gumbel(size=len(logw)), -np.inf)
    return np.argsort(-keys)[:m]
