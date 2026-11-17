"""Fixed held-out evaluation stream and snapshot scoring."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.simulator import simulate
from src.simulator.loop import DecisionPolicy, Decisions, EpochBatch, RulesPolicy
from src.training.evaluation import recall_at_fpr, value_detection_rate


class RecorderPolicy(DecisionPolicy):
    """Rules policy that records every decided frame (pre-decision columns + hist_* features)."""
    name = "recorder"

    def __init__(self):
        self.inner = RulesPolicy()
        self.frames: List[pd.DataFrame] = []

    def decide(self, batch: EpochBatch) -> Decisions:
        f = batch.frame.copy()
        f["_day"] = batch.day
        self.frames.append(f)
        return self.inner.decide(batch)


def build_eval_stream(cfg) -> pd.DataFrame:
    """One fixed world (its own seed) decided by the rules engine; every row carries its causal features and,
    for scoring only, the truth columns. All arms' models are scored on exactly this stream."""
    rec = RecorderPolicy()
    res = simulate(cfg, policy=rec)
    fr = pd.concat(rec.frames, ignore_index=True)
    tr = res.transactions[["transaction_id", "is_fraud_true", "fraud_scenario"]]
    ev = fr.merge(tr, on="transaction_id", how="left").rename(columns={"_day": "day"})
    ev["is_fraud_true"] = ev["is_fraud_true"].fillna(0).astype(int)
    ev["amount_usd"] = pd.to_numeric(ev["amount_usd"], errors="coerce").astype("float64")
    return ev


def metrics(y: np.ndarray, p: np.ndarray, amount: np.ndarray, fpr: float = 0.01) -> Dict:
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return dict(n=int(len(y)), n_fraud=int(y.sum()), auc_pr=float("nan"), recall_at_1pct_fpr=float("nan"),
                    vdr_at_1pct_fpr=float("nan"))
    return dict(n=int(len(y)), n_fraud=int(y.sum()), auc_pr=float(average_precision_score(y, p)),
                recall_at_1pct_fpr=recall_at_fpr(y, p, fpr), vdr_at_1pct_fpr=value_detection_rate(y, p, amount, fpr))


def score_snapshots(policy, ev: pd.DataFrame, fixed_days: tuple, period_days: int, n_days: int,
                    exclude_novel: bool = True) -> Dict:
    """For each snapshot: metrics on the fixed window. Also, per period, the metrics of the model that was in
    force at the start of the period (time-aligned AUC-PR / recall of the DEPLOYED model on the fixed stream)."""
    if not policy.snapshots:
        return dict(snapshots=[], aligned=[])
    X = policy.fz.transform(ev)
    novel = (ev["fraud_scenario"].to_numpy() == "bust_out") if exclude_novel else np.zeros(len(ev), bool)
    day = ev["day"].to_numpy()
    y = ev["is_fraud_true"].to_numpy()
    amt = ev["amount_usd"].fillna(0).to_numpy()
    P = [s["model"].predict_proba(X)[:, 1] for s in policy.snapshots]
    fw = (day >= fixed_days[0]) & (day < fixed_days[1]) & ~novel
    snaps = []
    for s, p in zip(policy.snapshots, P):
        m = metrics(y[fw], p[fw], amt[fw])
        m["day"] = s["day"]
        snaps.append(m)
    aligned = []
    start_days = np.array([s["day"] for s in policy.snapshots])
    for p0 in range(0, n_days, period_days):
        k = np.searchsorted(start_days, p0, side="right") - 1
        if k < 0:
            continue
        w = (day >= p0) & (day < p0 + period_days) & ~novel
        if w.sum() == 0:
            continue
        m = metrics(y[w], P[k][w], amt[w])
        m.update(day0=p0, model_day=int(start_days[k]))
        aligned.append(m)
    return dict(snapshots=snaps, aligned=aligned)
