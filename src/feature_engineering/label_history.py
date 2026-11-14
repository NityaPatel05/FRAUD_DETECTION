"""
Delayed label-history ("risk") features.

General fraud practice: fraud clusters on clients, cards, devices and addresses, so
the past fraud rate of the entities a transaction touches is one of the strongest
signals. Fraud labels (chargebacks) arrive late, so a row at time t may only use
labels of rows with time <= t - label_delay. That makes the feature identical
offline and online and keeps the row's own label out of its features.

For every key we emit:
  {key}_lbl_n      — # earlier rows of this entity with a known label
  {key}_lbl_fraud  — # of those that were fraud
  {key}_lbl_rate   — Bayesian-smoothed fraud rate (pulled to the global prior)
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from src.feature_engineering.entity_ops import make_codes, delayed_label_stats, smoothed_rate

LABEL_KEYS = ["uid1", "uid2", "card1", "card1_addr1", "addr1", "device_fp", "card2_dist1"]


def add_label_history(
    df: pd.DataFrame,
    t: np.ndarray,
    y: np.ndarray,
    delay_days: int,
    smoothing: float,
    prior_rate: float,
    keys: Sequence[str] = LABEL_KEYS,
) -> Dict[str, np.ndarray]:
    """``y``: isFraud with NaN for unlabelled rows (Kaggle test)."""
    out: Dict[str, np.ndarray] = {}
    delay = int(delay_days * 86400)
    for k in keys:
        if k not in df.columns:
            continue
        codes = make_codes(df, k)
        st = delayed_label_stats(codes, t, y, delay)
        out[f"{k}_lbl_n"] = st["n"].astype(np.float32)
        out[f"{k}_lbl_fraud"] = st["fraud"].astype(np.float32)
        out[f"{k}_lbl_rate"] = smoothed_rate(st["fraud"], st["n"], prior_rate, smoothing).astype(np.float32)
    logger.info("Label-history features (delay={}d, smoothing={}): {} columns.",
                delay_days, smoothing, len(out))
    return out
