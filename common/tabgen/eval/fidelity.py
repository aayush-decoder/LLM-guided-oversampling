"""
Quality / diversity / privacy metrics of Section IV-A2 and IV-C2, all in the encoded space.

discriminator  (lower better)  XGBoost real-vs-synthetic on a held-out split. score = 2*max(AUC,.5)-1,
                               so 0 = indistinguishable, 1 = perfectly separable (paper convention).
inverse_kl     (higher better) 1/M * sum_i 1/(1+KL(p(X_i)||p(^X_i)))  (synthcity-style, 10-bin hist).
density        (higher better) Naeem et al. 2020, k=2.
coverage       (higher better) Naeem et al. 2020, k=2.
dcr            distance of every real sample to its closest synthetic sample (Fig. 6).
"""

from typing import Any, Optional, List, Dict, Union

def _subsample(a, n, rng):
    pass

def discriminator_score(real, fake, seed):
    pass

def inverse_kl(real, fake, cols, discrete, bins):
    pass

def density_coverage(real, fake, k):
    pass

def dcr(reference, query):
    """For every row of `query`, Euclidean distance to its nearest row in `reference`."""
    pass

def fidelity_report(real, fake, features, discrete, seed, max_rows):
    pass
