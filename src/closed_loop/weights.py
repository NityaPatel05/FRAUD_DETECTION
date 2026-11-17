"""Inverse-propensity weights for a policy that explores."""
from __future__ import annotations

import numpy as np


def ipw_weights(propensity: np.ndarray, max_weight: float = 100.0) -> np.ndarray:
    """w = 1 / P(action taken | features), clipped at ``max_weight``.

    A would-be decline that the engine let through with probability eps carries propensity eps, so its weight is
    1/eps: it stands in for the 1/eps similar rows that were declined and never labelled. Rows the policy
    approved on its own have propensity 1 and weight 1. Missing / non-positive propensities count as 1."""
    p = np.asarray(propensity, dtype=float)
    p = np.where(np.isfinite(p) & (p > 0), p, 1.0)
    return np.minimum(1.0 / p, max_weight)
