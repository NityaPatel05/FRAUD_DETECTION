from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def build_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive log, decimal, and deviation features from TransactionAmt.

    Requires uid aggregation columns (uid_amt_mean, uid_amt_std, uid_amt_max,
    card1_amt_mean) to already exist in df.
    """
    # ── Log transform (right-skewed → approx normal) ─────────────────────────
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # ── Decimal / cent part  (fraud clusters at unusual cent values) ──────────
    df["TransactionAmt_decimal"] = (
        (df["TransactionAmt"] - df["TransactionAmt"].astype(int)).round(2)
    )

    # ── Deviation from uid / card1 mean ──────────────────────────────────────
    eps = 1e-5

    if "uid1_amt_mean" in df.columns:
        df["amt_over_uid_mean"] = (
            df["TransactionAmt"] / (df["uid1_amt_mean"].abs() + eps)
        )
        df["amt_over_uid_max"] = (
            df["TransactionAmt"] / (df["uid1_amt_max"].abs() + eps)
        )

    if "card1_amt_mean" in df.columns:
        df["amt_over_card1_mean"] = (
            df["TransactionAmt"] / (df["card1_amt_mean"].abs() + eps)
        )

    # ── Z-score: (amt - uid_mean) / (uid_std + ε) ────────────────────────────
    # Direct anomaly score — fed to VAE as input feature
    if "uid1_amt_mean" in df.columns and "uid1_amt_std" in df.columns:
        df["amt_zscore"] = (
            (df["TransactionAmt"] - df["uid1_amt_mean"])
            / (df["uid1_amt_std"].abs() + eps)
        )
    else:
        # Fallback: global zscore if uid aggregations not yet available
        mu  = df["TransactionAmt"].mean()
        std = df["TransactionAmt"].std()
        df["amt_zscore"] = (df["TransactionAmt"] - mu) / (std + eps)
        logger.warning(
            "uid1_amt_mean not found — using global zscore for amt_zscore. "
            "Run uid_aggregations.py first for correct computation."
        )

    logger.info(
        "Amount features added: log, decimal, amt_over_uid_mean, "
        "amt_over_uid_max, amt_over_card1_mean, amt_zscore."
    )
    return df
