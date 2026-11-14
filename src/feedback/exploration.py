"""
Counterfactual evaluation with propensity-logged exploration (Stripe; Manapat, 2018).

Once a model blocks transactions you never learn whether the blocked ones were fraud, so
naive metrics on observed outcomes are biased, and each retrain learns only what the
current model lets through. Fix: let a SMALL, score-dependent fraction of would-be
blocks through, log P(allowed), and weight observed outcomes by 1 / P(allowed)
(inverse propensity) to estimate the metrics of ANY candidate policy.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def allow_probability(p: np.ndarray, block: np.ndarray, q_max: float = 0.05, q_min: float = 0.001,
                      steepness: float = 5.0) -> np.ndarray:
    """P(transaction is let through): 1 if not blocked; for blocked, decays with the score
    (riskiest payments are explored least, as in Stripe's scheme)."""
    q = np.clip(q_max * np.exp(-steepness * np.asarray(p)), q_min, q_max)
    return np.where(block, q, 1.0)


def ips_policy_metrics(y: np.ndarray, observed: np.ndarray, propensity: np.ndarray,
                       candidate_block: np.ndarray) -> Dict[str, float]:
    """Inverse-propensity estimates of recall / precision / block-rate of a candidate policy.

    y: labels (used only where observed); observed: label was learnt (txn was allowed);
    propensity: P(allowed) when logged; candidate_block: what the NEW policy would block.
    """
    w = np.where(observed, 1.0 / propensity, 0.0)
    yy = np.where(observed, y, 0.0)
    fraud_total = np.sum(w * yy)
    caught = np.sum(w * yy * candidate_block)
    blocked = np.sum(w * candidate_block)
    return {"recall": float(caught / fraud_total) if fraud_total else float("nan"),
            "precision": float(caught / blocked) if blocked else float("nan"),
            "block_rate": float(blocked / np.sum(w))}


def naive_policy_metrics(y: np.ndarray, observed: np.ndarray, candidate_block: np.ndarray) -> Dict[str, float]:
    """What a team without propensity logging computes (biased: blocked fraud is invisible)."""
    yy, b = y[observed], candidate_block[observed]
    return {"recall": float((yy * b).sum() / max(yy.sum(), 1)),
            "precision": float((yy * b).sum() / max(b.sum(), 1)),
            "block_rate": float(b.mean())}
