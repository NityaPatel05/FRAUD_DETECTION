"""
Delayed label-history ("risk") features under the shared label policy.

General fraud practice: fraud clusters on clients, cards, devices and addresses, so
the past fraud rate of the entities a transaction touches is one of the strongest
signals. Labels are late: a fraud becomes known ``d`` days after its transaction
(per-transaction delay from ``src.labels.policy``) and a transaction with no report is
only trusted as legit once it is ``maturity_days`` old. A row at time t may use only what
was known at t, which makes the feature identical offline and online and keeps the row's
own label out of its features.

For every key we emit:
  {key}_lbl_n      -- # earlier rows of this entity that are mature
  {key}_lbl_fraud  -- # earlier rows that are known fraud (any age: fresh chargebacks count)
  {key}_lbl_rate   -- smoothed (fraud among mature rows) / (mature rows), pulled to the prior
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from src.feature_engineering.entity_ops import make_codes, policy_label_stats, smoothed_rate
from src.labels.policy import LabelEvents

LABEL_KEYS = ["uid1", "uid2", "card1", "card1_addr1", "addr1", "device_fp", "card2_dist1"]


def add_label_history(
    df: pd.DataFrame,
    t: np.ndarray,
    events: LabelEvents,
    smoothing: float,
    prior_rate: float,
    keys: Sequence[str] = LABEL_KEYS,
) -> Dict[str, np.ndarray]:
    """``events``: ``label_ready_times(y, TransactionID, t, policy)`` (unlabelled rows never count)."""
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        if k not in df.columns:
            continue
        codes = make_codes(df, k)
        st = policy_label_stats(codes, t, events)
        out[f"{k}_lbl_n"] = st["n"].astype(np.float32)
        out[f"{k}_lbl_fraud"] = st["fraud"].astype(np.float32)
        out[f"{k}_lbl_rate"] = smoothed_rate(st["fraud_mature"], st["n"], prior_rate, smoothing).astype(np.float32)
    logger.info("Label-history features (smoothing={}): {} columns.", smoothing, len(out))
    return out
