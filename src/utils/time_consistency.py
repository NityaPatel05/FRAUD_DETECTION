"""
Time-consistency feature check (replaces the adversarial "drift" filter).

For each feature on its own: train a tiny LightGBM on the FIRST month of the
training period and score it on the LAST month. A feature whose relationship with
fraud does not survive over time (val AUC <= threshold) is noise or leaks the
period — drop it. Drift itself is not a reason to drop a feature: many drifting
features (amount, card ids, counts) are still highly predictive.

IEEE-CIS 1st place used exactly this check (dropped 19 features).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score


def time_consistency_check(
    df: pd.DataFrame,
    features: List[str],
    window_days: int = 30,
    threshold: float = 0.5,
    train_start_day: int | None = None,
    train_end_day: int | None = None,
) -> pd.DataFrame:
    """train window = [train_start_day, train_end_day) (default: first `window_days` days)."""
    import lightgbm as lgb

    day = df["day"].to_numpy()
    lo = day.min() if train_start_day is None else train_start_day
    hi = (lo + window_days) if train_end_day is None else train_end_day
    first = (day >= lo) & (day < hi)
    last = day > day.max() - window_days
    y = df["isFraud"].to_numpy().astype(int)
    rows = []
    for i, f in enumerate(features):
        x = df[[f]]
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=8,
                               min_child_samples=200, verbose=-1, n_jobs=-1)
        m.fit(x[first], y[first])
        auc_tr = roc_auc_score(y[first], m.predict_proba(x[first])[:, 1])
        auc_va = roc_auc_score(y[last], m.predict_proba(x[last])[:, 1])
        rows.append({"feature": f, "auc_first_month": auc_tr, "auc_last_month": auc_va})
        if (i + 1) % 50 == 0:
            logger.info("time-consistency: {}/{} features done", i + 1, len(features))
    rep = pd.DataFrame(rows)
    rep["drop"] = rep["auc_last_month"] <= threshold
    logger.info("Time-consistency: {} of {} features fail (last-month AUC <= {}).",
                int(rep["drop"].sum()), len(rep), threshold)
    return rep.sort_values("auc_last_month")
