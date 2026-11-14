from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from loguru import logger


# ─────────────────────────────────────────────────────────
# 1.  Load & merge raw CSVs
# ─────────────────────────────────────────────────────────

def load_raw_data(raw_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge train/test transaction + identity tables.

    Returns
    -------
    train_df, test_df — merged, NOT yet sorted.
    """
    raw = Path(raw_dir)
    logger.info("Loading raw CSVs from {}", raw)

    train_txn = pd.read_csv(raw / "train_transaction.csv")
    train_id  = pd.read_csv(raw / "train_identity.csv")
    test_txn  = pd.read_csv(raw / "test_transaction.csv")
    test_id   = pd.read_csv(raw / "test_identity.csv")
    # Kaggle's test_identity.csv names columns "id-01".."id-38" while train uses
    # "id_01".."id_38". Without this rename every identity feature is missing on test.
    test_id.columns = [c.replace("id-", "id_") for c in test_id.columns]

    train_df = train_txn.merge(train_id, on="TransactionID", how="left")
    test_df  = test_txn.merge(test_id,  on="TransactionID", how="left")

    logger.info("Train shape: {}  |  Test shape: {}", train_df.shape, test_df.shape)
    return train_df, test_df


# ─────────────────────────────────────────────────────────
# 2.  Sort by TransactionDT — THE FIRST THING WE DO
# ─────────────────────────────────────────────────────────

def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort ascending by TransactionDT and reset index.
    Must be called before ANY feature creation, encoding, or split.
    Derive day / hour columns here so they are available for everything else.
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # TransactionDT // 86400 = day number (0-based from reference epoch)
    df["day"]  = df["TransactionDT"] // 86400
    # hour of day  0-23
    df["hour"] = (df["TransactionDT"] // 3600) % 24

    logger.info(
        "Sorted by TransactionDT.  Day range: {} – {}  |  "
        "Hour distribution: {:.1f}±{:.1f}",
        df["day"].min(), df["day"].max(),
        df["hour"].mean(), df["hour"].std(),
    )
    return df


# ─────────────────────────────────────────────────────────
# 3.  Time-aware train / validation split
# ─────────────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    train_quantile: float = 0.75,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split on the 75th-percentile day boundary.

    NEVER use random_state / shuffle — this leaks future into past.

    Returns
    -------
    train_df, val_df
    """
    split_day = df["day"].quantile(train_quantile)
    train = df[df["day"] <= split_day].copy()
    val   = df[df["day"] >  split_day].copy()

    logger.info(
        "Time split at day {} (q={}).  "
        "Train: {} rows ({} fraud)  |  Val: {} rows ({} fraud)",
        int(split_day), train_quantile,
        len(train), int(train["isFraud"].sum()),
        len(val),   int(val["isFraud"].sum()),
    )
    return train, val


# ─────────────────────────────────────────────────────────
# 4.  Expanding-window CV fold generator
# ─────────────────────────────────────────────────────────

def expanding_window_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    min_train_days: int = 30,
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, val_idx) for expanding-window time-series CV.

    Fold k: train = days [0 .. cutoff_k],  val = days (cutoff_k .. cutoff_{k+1}]

    Never shuffles.  Val window is always strictly after train.

    Example with n_folds=5, day range 0-150:
      Fold 0: train day 0-60,  val day 61-90
      Fold 1: train day 0-90,  val day 91-120
      Fold 2: train day 0-120, val day 121-150
    """
    days = df["day"].values
    day_min, day_max = days.min(), days.max()

    # Split day-range into n_folds+1 equal segments
    cut_points = np.linspace(day_min + min_train_days, day_max, n_folds + 1)

    folds = []
    for i in range(n_folds):
        train_end = cut_points[i]
        val_end   = cut_points[i + 1]
        train_idx = np.where(days <= train_end)[0]
        val_idx   = np.where((days > train_end) & (days <= val_end))[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        folds.append((train_idx, val_idx))
        logger.debug(
            "Fold {}: train rows={} (day ≤{:.0f})  val rows={} (day {:.0f}–{:.0f})",
            i, len(train_idx), train_end,
            len(val_idx), train_end, val_end,
        )

    logger.info("Generated {} expanding-window CV folds", len(folds))
    return folds
