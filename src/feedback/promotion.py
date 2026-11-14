"""
Champion / challenger promotion gate (Stripe blue-green, Nubank shadow tests, Sift SafetyNet).

A challenger scores live traffic in SHADOW (logged only). It is promoted only if, on the
same traffic with MATURED labels:
  1. it is better on the metrics that matter (AUC-PR and recall at a fixed FPR), and
  2. at the CURRENT decision thresholds its decline / review / step-up rates stay within
     tolerance of the champion's — a better AUC can still flood the review queue.
quantile_map() can align challenger scores to the champion's distribution so existing
thresholds keep the same action rates (Sift's approach).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from sklearn.metrics import average_precision_score

from src.training.evaluation import recall_at_fpr


def quantile_map(challenger_ref: np.ndarray, champion_ref: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return f(score) mapping challenger scores onto the champion's score distribution."""
    q = np.linspace(0, 1, 1001)
    cq, hq = np.quantile(challenger_ref, q), np.quantile(champion_ref, q)
    cq = np.maximum.accumulate(cq + np.arange(len(cq)) * 1e-12)       # strictly increasing for interp
    return lambda s: np.interp(s, cq, hq)


def action_rates(actions: np.ndarray) -> Dict[str, float]:
    vals, cnt = np.unique(actions, return_counts=True)
    return {str(v): float(c) / len(actions) for v, c in zip(vals, cnt)}


def promotion_gate(y: np.ndarray, p_champion: np.ndarray, p_challenger: np.ndarray,
                   act_champion: np.ndarray, act_challenger: np.ndarray,
                   min_ap_gain: float = 0.005, max_rate_shift: float = 0.25, fpr: float = 0.01) -> Dict:
    """max_rate_shift: allowed RELATIVE change of each non-approve action rate (0.25 = ±25%)."""
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(y)
    y_, a, b = y[m].astype(int), np.asarray(p_champion)[m], np.asarray(p_challenger)[m]
    res = {"n_matured": int(m.sum()),
           "ap_champion": float(average_precision_score(y_, a)),
           "ap_challenger": float(average_precision_score(y_, b)),
           "recall_fpr_champion": recall_at_fpr(y_, a, fpr),
           "recall_fpr_challenger": recall_at_fpr(y_, b, fpr)}
    rc, rn = action_rates(np.asarray(act_champion)), action_rates(np.asarray(act_challenger))
    shifts = {k: (rn.get(k, 0.0) - rc.get(k, 0.0)) / max(rc.get(k, 0.0), 1e-4)
              for k in set(rc) | set(rn) if k != "approve"}
    res["action_rates_champion"], res["action_rates_challenger"], res["relative_rate_shift"] = rc, rn, shifts
    reasons = []
    gain = res["ap_challenger"] - res["ap_champion"]
    if gain < min_ap_gain:
        reasons.append(f"AUC-PR gain {gain:+.4f} < {min_ap_gain}")
    if res["recall_fpr_challenger"] < res["recall_fpr_champion"]:
        reasons.append("recall at fixed FPR got worse")
    for k, v in shifts.items():
        if abs(v) > max_rate_shift:
            reasons.append(f"{k} rate shifts {v:+.0%} at current thresholds (SafetyNet)")
    res["promote"] = not reasons
    res["reasons"] = reasons or ["all checks passed"]
    return res
