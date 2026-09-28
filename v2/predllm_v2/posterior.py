"""
Label querying via a permutation-marginalised exact posterior (v2 novelty N4) - LLM USED HERE.

Classification: for R feature orders pi_1..pi_R (pi_1 = original order, the rest random) score every
class exactly:
        p_r(y=c | x) = p(" c" EOS | pi_r(x), "target is") / sum_c' p(" c'" EOS | ...)
        p_hat(y | x) = 1/R * sum_r p_r(y | x)            (mixture over orders)
then  y ~ p_hat^(1/T_y)  (label_decode=sample)  or  y = argmax p_hat  (label_decode=argmax).

Regression: y_r ~ FSM-constrained decode after pi_r(x); y = mean_r y_r (reg_aggregate=mean) or y_1.
"""

from typing import Any, Optional, List, Dict, Union

def _orders(features, R, rng):
    pass

def class_posterior(model, codec, rows, R, cfg, rng):
    pass

def decide_labels(probs, how, temperature, rng):
    pass

def regression_values(model, codec, rows, R, cfg, cons, gen, rng, empirical):
    pass
