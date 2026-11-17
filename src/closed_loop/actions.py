"""Action rules: map a fraud probability (and the row) to approve / step_up_3ds / review / decline."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import COST_DEFAULTS

ACTION_ORDER = ("approve", "step_up_3ds", "review", "decline")


def can_step_up(frame: pd.DataFrame) -> np.ndarray:
    return (frame["channel"].to_numpy() == "CNP") & (frame["three_ds_result"].to_numpy() == "not_used")


def segment_of(frame: pd.DataFrame) -> np.ndarray:
    """channel x (new vs established customer): e.g. 'CNP|new'. New = guest or account younger than 30 days."""
    age = pd.to_numeric(frame["account_age_days"], errors="coerce").astype("float64").to_numpy()
    new = frame["customer_id"].isna().to_numpy() | (np.nan_to_num(age, nan=0.0) < 30)
    return np.char.add(np.char.add(frame["channel"].astype(str).to_numpy().astype("U8"), "|"),
                       np.where(new, "new", "est"))


def expected_costs(p: np.ndarray, amount: np.ndarray, step_ok: np.ndarray, cost: Dict | None = None) -> Dict[str, np.ndarray]:
    """Expected USD cost of every action for every row, given P(fraud)=p. Unavailable actions are +inf."""
    c = dict(COST_DEFAULTS, **(cost or {}))
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    amt = np.asarray(amount, float)
    loss = c["fraud_loss_frac"] * amt + c["chargeback_fee"]            # what an approved fraud costs
    margin = c["margin_rate"] * amt
    fd = margin + c["churn_cost"]                                       # what a declined legit customer costs
    out = {
        "approve": p * loss,
        "decline": (1 - p) * fd,
        "step_up_3ds": (p * c["stepup_fraud_pass"] * (1 - c["liability_shift_share"]) * loss
                        + (1 - p) * (1 - c["stepup_legit_pass"]) * (margin + c["stepup_churn_frac"] * c["churn_cost"])
                        + c["stepup_fee"]),
        "review": (c["review_cost"] + p * (1 - c["review_accuracy"]) * loss
                   + (1 - p) * (c["review_false_decline"] * fd + c["review_abandon"] * margin)),
    }
    out["step_up_3ds"] = np.where(step_ok, out["step_up_3ds"], np.inf)
    return out


def choose_actions(costs: Dict[str, np.ndarray], review_capacity: int) -> np.ndarray:
    """Argmin expected cost per row; at most ``review_capacity`` rows go to review (largest saving first, the
    rest fall back to their best non-review action)."""
    names = [a for a in ACTION_ORDER if a != "review"]
    M = np.stack([costs[a] for a in names])
    best_i = M.argmin(axis=0)
    best = M.min(axis=0)
    action = np.array(names, dtype=object)[best_i]
    saving = best - costs["review"]
    cand = np.nonzero(saving > 0)[0]
    if len(cand) and review_capacity > 0:
        top = cand[np.argsort(-saving[cand], kind="stable")[:review_capacity]]
        action[top] = "review"
    return action


class ThresholdRule:
    """Decline at p >= t, approve otherwise (the single-threshold baseline)."""
    kind = "threshold"

    def __init__(self, t: float = 0.30):
        self.t = float(t)

    def decide(self, frame: pd.DataFrame, p: np.ndarray) -> np.ndarray:
        return np.where(p >= self.t, "decline", "approve").astype(object)


class SegmentThresholdRule:
    """One decline threshold per segment (channel x new/established customer)."""
    kind = "segment_threshold"

    def __init__(self, thresholds: Dict[str, float], default: float = 0.30):
        self.thresholds, self.default = dict(thresholds), float(default)

    def decide(self, frame: pd.DataFrame, p: np.ndarray) -> np.ndarray:
        seg = segment_of(frame)
        t = np.array([self.thresholds.get(s, self.default) for s in seg])
        return np.where(p >= t, "decline", "approve").astype(object)

    def fit(self, p, y, w, amount, seg, cost=None, segmented: bool = True, min_pos: int = 8) -> None:
        """Tune on held-out scored rows. segmented=False tunes ONE global threshold (the single-threshold baseline)."""
        g = tune_threshold(p, y, w, amount, None, cost, min_pos=min_pos, fallback=self.default)
        self.default = g
        self.thresholds = {}
        if segmented:
            for s in np.unique(seg):
                self.thresholds[str(s)] = tune_threshold(p, y, w, amount, seg == s, cost, min_pos=min_pos, fallback=g)


class CostRule:
    """Expected-cost action selection over approve / step_up_3ds / review / decline."""
    kind = "expected_cost"

    def __init__(self, cost: Dict | None = None):
        self.cost = dict(COST_DEFAULTS, **(cost or {}))

    def decide(self, frame: pd.DataFrame, p: np.ndarray) -> np.ndarray:
        amt = pd.to_numeric(frame["amount_usd"], errors="coerce").astype("float64").to_numpy()
        costs = expected_costs(p, np.nan_to_num(amt), can_step_up(frame), self.cost)
        cap = int(np.ceil(self.cost["review_capacity_frac"] * len(frame)))
        return choose_actions(costs, cap)


def tune_threshold(p: np.ndarray, y: np.ndarray, w: np.ndarray, amount: np.ndarray, mask: Optional[np.ndarray] = None,
                   cost: Dict | None = None, min_pos: int = 8, fallback: float = 0.30) -> float:
    """Threshold minimising  sum_w[y=1,p<t] * fraud_loss + sum_w[y=0,p>=t] * false_decline_cost  on held-out rows."""
    c = dict(COST_DEFAULTS, **(cost or {}))
    m = np.ones(len(p), bool) if mask is None else mask
    if m.sum() == 0 or (y[m] == 1).sum() < min_pos:
        return fallback
    p, y, w, a = p[m], y[m], w[m], np.asarray(amount, float)[m]
    loss = (c["fraud_loss_frac"] * a + c["chargeback_fee"]) * w
    fd = (c["margin_rate"] * a + c["churn_cost"]) * w
    cand = np.unique(np.concatenate([np.quantile(p, np.linspace(0.90, 0.9995, 60)), np.linspace(0.05, 0.9, 18)]))
    best_t, best = fallback, np.inf
    order = np.argsort(p)
    ps, ls, fs, ys = p[order], loss[order], fd[order], y[order]
    cl = np.concatenate([[0.0], np.cumsum(ls * (ys == 1))])         # fraud loss of rows below the threshold
    cf = np.concatenate([[0.0], np.cumsum(fs * (ys == 0))])
    for t in cand:
        i = np.searchsorted(ps, t, side="left")
        tot = cl[i] + (cf[-1] - cf[i])
        if tot < best:
            best, best_t = tot, float(t)
    return float(np.clip(best_t, 0.02, 0.98))
