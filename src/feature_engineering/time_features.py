from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build cyclic hour / day-of-week / week / month features plus
    boolean convenience flags.

    Requires 'TransactionDT' and 'day' columns (from sort_and_split).
    """
    # ── Raw extractions ───────────────────────────────────────────────────────
    df["hour"]      = (df["TransactionDT"] // 3600) % 24
    df["day"]       = df["TransactionDT"] // 86400
    df["weekday"]   = df["day"] % 7
    df["week"]      = df["day"] // 7

    df["is_weekend"]   = (df["weekday"] >= 5).astype(int)
    df["is_late_night"] = (
        ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    )

    # ── Cyclic encoding — ACM 2025 method ────────────────────────────────────
    # sin/cos encoding makes period endpoints adjacent in feature space

    # Hour (period = 24)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)

    # Day of week (period = 7)
    df["dow_sin"]     = np.sin(2 * np.pi * df["weekday"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["weekday"] / 7)

    # Week (period = 52)
    df["week_sin"]    = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"]    = np.cos(2 * np.pi * df["week"] / 52)

    # ── Month extraction (for GroupKFold CV) ─────────────────────────────────
    # TransactionDT is seconds from a reference point.
    # Approx month: day // 30  (good enough for fold assignment)
    df["month"] = df["day"] // 30

    logger.info(
        "Time features added: hour, weekday, week, is_weekend, is_late_night, "
        "hour_sin/cos, dow_sin/cos, week_sin/cos, month."
    )
    return df
