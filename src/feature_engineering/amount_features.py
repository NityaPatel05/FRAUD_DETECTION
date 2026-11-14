from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def build_amount_features(df: pd.DataFrame, uid_col: str | None = "uid1") -> pd.DataFrame:
    """
    Derive log, decimal, and deviation features from TransactionAmt.

    amt_zscore is computed with a CAUSAL expanding window using .shift(1) so
    each transaction's z-score only uses strictly prior transactions of that
    user. Requires df to be sorted by [TransactionDT, TransactionID] before
    calling (a sort guard is applied internally as well).

    All other features (amt_over_uid_mean, etc.) still use the static uid
    aggregation columns produced by uid_aggregations.py — these are safe for
    tree models where train_idx masking handles train→test leakage.
    """
    # ── Log transform (right-skewed → approx normal) ─────────────────────────
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # ── Decimal / cent part  (fraud clusters at unusual cent values) ──────────
    df["TransactionAmt_decimal"] = (
        (df["TransactionAmt"] - df["TransactionAmt"].astype(int)).round(2)
    )

    # ── Deviation from uid / card1 mean ──────────────────────────────────────
    eps = 1e-5

    if f"{uid_col}_amt_mean" in df.columns:
        df["amt_over_uid_mean"] = (
            df["TransactionAmt"] / (df[f"{uid_col}_amt_mean"].abs() + eps)
        )
        df["amt_over_uid_max"] = (
            df["TransactionAmt"] / (df[f"{uid_col}_amt_max"].abs() + eps)
        )

    if "card1_amt_mean" in df.columns:
        df["amt_over_card1_mean"] = (
            df["TransactionAmt"] / (df["card1_amt_mean"].abs() + eps)
        )

    # ── Causal Expanding Z-score: (amt - past_mean) / (past_std + ε) ──────────
    # .shift(1) is the KEY: current row only uses statistics from PRIOR rows.
    # → First transaction per user: past_mean=NaN, past_std=NaN → z-score=0.0
    # → Second transaction: past_mean = first tx amount, past_std = NaN → 0.0
    # → Third+: fully causal — only past transactions contribute.
    if uid_col and uid_col in df.columns:
        # Sort by [uid, time] — uid must be primary key so that .transform().shift(1)
        # stays within each user's ordered block and never crosses user boundaries.
        sort_cols = [uid_col] + [c for c in ["TransactionDT", "TransactionID"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        causal_mean = df.groupby(uid_col)["TransactionAmt"].transform(
            lambda s: s.expanding().mean().shift(1)
        )
        causal_std = df.groupby(uid_col)["TransactionAmt"].transform(
            lambda s: s.expanding().std().shift(1)
        )

        df["amt_zscore"] = (
            (df["TransactionAmt"] - causal_mean) / (causal_std.abs() + eps)
        ).fillna(0.0)  # NaN on first (and second for std) tx per user → z=0 (no history)

        logger.info(
            "amt_zscore: causal expanding window (shift=1) on uid_col='%s'. "
            "No future leakage. First-tx rows → 0.0.", uid_col
        )
    else:
        # Fallback: global zscore
        mu  = df["TransactionAmt"].mean()
        std = df["TransactionAmt"].std()
        df["amt_zscore"] = (df["TransactionAmt"] - mu) / (std + eps)
        logger.warning(
            "uid_col='%s' not present in df — falling back to global zscore. "
            "amt_zscore will contain future leakage. Ensure uid1 is built "
            "before build_amount_features() for BiLSTM use.", uid_col
        )

    logger.info(
        "Amount features added: log, decimal, amt_over_uid_mean, "
        "amt_over_uid_max, amt_over_card1_mean, amt_zscore."
    )
    return df
