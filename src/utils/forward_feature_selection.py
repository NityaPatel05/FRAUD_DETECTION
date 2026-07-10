from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from loguru import logger


# ── Quick LGBM for selection ──────────────────────────────────────────────────
SELECTION_PARAMS = {
    "n_estimators":    300,
    "num_leaves":      64,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "scale_pos_weight": 27,
    "metric":          "average_precision",
    "objective":       "binary",
    "random_state":    42,
    "verbose":         -1,
    "n_jobs":          -1,
}


def _quick_aupr(
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    feat_cols: List[str],
    target:    str = "isFraud",
) -> float:
    """Train a quick LightGBM and return val AUC-PR."""
    present = [c for c in feat_cols if c in train_df.columns and c in val_df.columns]
    if not present:
        return 0.0

    X_tr = train_df[present].fillna(-999).values
    y_tr = train_df[target].values
    X_vl = val_df[present].fillna(-999).values
    y_vl = val_df[target].values

    cb = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=9999),
    ]

    model = lgb.LGBMClassifier(**SELECTION_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=cb)

    preds = model.predict_proba(X_vl)[:, 1]
    return average_precision_score(y_vl, preds)


def forward_d_column_selection(
    train_df:       pd.DataFrame,
    val_df:         pd.DataFrame,
    baseline_cols:  List[str],
    candidate_cols: List[str],
    other_feats:    List[str] = None,
    min_improvement: float = 0.0005,
) -> Tuple[List[str], dict]:
    """
    Forward selection for D_norm columns.

    Parameters
    ----------
    train_df, val_df   : time-split dataframes
    baseline_cols      : starting feature set (e.g. ['D1_norm'])
    candidate_cols     : columns to evaluate one by one
    other_feats        : always-included features (e.g. uid_magic_freq, amt features)
    min_improvement    : minimum AUC-PR delta to keep a feature

    Returns
    -------
    (kept_cols, selection_log)
    """
    if other_feats is None:
        other_feats = ["uid1_freq", "uid_magic_freq", "TransactionAmt_log",
                       "card1_freq", "addr1_freq"]

    kept      = list(baseline_cols)
    base_feats = kept + other_feats
    baseline_aupr = _quick_aupr(train_df, val_df, base_feats)

    logger.info(
        "Forward D selection — Baseline ({}) AUC-PR: {:.5f}",
        baseline_cols, baseline_aupr,
    )

    log = {"baseline": {"features": baseline_cols, "aupr": baseline_aupr}}
    current_aupr = baseline_aupr

    for col in candidate_cols:
        if col in kept:
            continue

        trial_feats = kept + [col] + other_feats
        trial_aupr  = _quick_aupr(train_df, val_df, trial_feats)
        delta       = trial_aupr - current_aupr

        if delta > min_improvement:
            kept.append(col)
            current_aupr = trial_aupr
            status = "KEEP ✓"
        else:
            status = "DROP  ✗"

        logger.info(
            "  {:15s} AUC-PR: {:.5f}  (Δ={:+.5f})  {}",
            col, trial_aupr, delta, status,
        )
        log[col] = {"aupr": trial_aupr, "delta": delta, "kept": delta > min_improvement}

    logger.info(
        "Forward selection complete.  Kept {}/{} D columns: {}",
        len(kept), len(candidate_cols) + len(baseline_cols), kept,
    )
    return kept, log


def run_d_column_selection(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
) -> List[str]:
    """
    Convenience wrapper for D column forward selection.
    Run after D normalisation (d_features.py).
    Returns list of D_norm columns to keep.
    """
    baseline   = ["D1_norm"]
    candidates = [
        "D2_norm", "D3_norm", "D4_norm", "D5_norm",
        "D6_norm", "D7_norm", "D8_norm", "D9_norm",
        "D10_norm", "D11_norm", "D12_norm", "D13_norm",
        "D14_norm", "D15_norm",
    ]
    kept, _ = forward_d_column_selection(train_df, val_df, baseline, candidates)
    return kept
