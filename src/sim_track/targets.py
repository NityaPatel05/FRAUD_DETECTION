"""
Training targets AS KNOWN AT A CUTOFF for the simulated world.

Semantics (mirrors src/labels/policy.py, but with the world's real label dates instead of a hash delay):
  * a row's label exists only from its ``label_date`` on; at cutoff c
        target = 1  iff  reported_label == 1  and  label_date <= c
        target = 0  otherwise  (reported 0 / mature with no dispute, AND still-pending rows: a fraud that is
                    still in its chargeback delay is, exactly like in production, a (wrong) negative)
  * usable rows: event_time < c - gap_days (the "delay period") and the row has an outcome at all.
        ``label_date`` null == label_status 'unobserved' (declined / never observed): no outcome, so the row is
        EXCLUDED from training. This is the real selection bias of a closed-loop system (studied in Phase C2).
  * ``is_fraud_true`` is never read here. Nothing after ``c`` is read except to decide "label_date <= c".
Folds are rolling:  train | gap | val(es + score);  early stopping uses only the first part of the val window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

DAY = 86400


def known_target(reported_label: pd.Series, label_date: pd.Series, event_time: pd.Series, cutoff: pd.Timestamp,
                 gap_days: float = 14.0) -> Tuple[np.ndarray, np.ndarray]:
    """(target 0/1, usable mask) a model deployed at ``cutoff`` could have been trained on."""
    rep = reported_label.fillna(0).to_numpy().astype(np.int64)
    ld = label_date
    arrived = (ld.notna() & (ld <= cutoff)).to_numpy()
    target = ((rep == 1) & arrived).astype(np.int8)
    usable = ((event_time < cutoff - pd.Timedelta(days=gap_days)) & ld.notna()).to_numpy()
    return target, usable


@dataclass
class SimFold:
    k: int
    train: np.ndarray
    y_train: np.ndarray
    es: np.ndarray
    score: np.ndarray
    cutoff: pd.Timestamp


def make_sim_folds(df: pd.DataFrame, period_end: pd.Timestamp, n_folds: int, val_days: int, gap_days: float,
                   es_frac: float = 0.5) -> List[SimFold]:
    """Walk-forward folds tiling the last n_folds*val_days days before ``period_end``."""
    et = df["event_time"]
    es_days = int(np.clip(round(val_days * es_frac), 0, val_days - 1))
    folds: List[SimFold] = []
    for k in range(n_folds):
        v0 = period_end - pd.Timedelta(days=(n_folds - k) * val_days)
        v1 = v0 + pd.Timedelta(days=val_days)
        y, usable = known_target(df["reported_label"], df["label_date"], et, v0, gap_days)
        usable &= (et < v0).to_numpy()
        tr = np.where(usable)[0]
        es = np.where(((et >= v0) & (et < v0 + pd.Timedelta(days=es_days))).to_numpy())[0]
        sc = np.where(((et >= v0 + pd.Timedelta(days=es_days)) & (et < v1)).to_numpy())[0]
        if len(tr) and len(sc):
            folds.append(SimFold(k, tr, y[tr].astype(int), es if len(es) else sc, sc, v0))
    return folds
