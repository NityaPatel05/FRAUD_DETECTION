from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger


C_COLS = ["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14"]

# Explicitly named as top features in top-5% Kaggle solution
PRIORITY_C_AGGS = ["C1","C9","C11","C13"]


def build_c_features(
    df: pd.DataFrame,
    train_idx: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    1. Add ratio features (C1/C2, C5/C6)
    2. Add C3 fraud flag (C3 ≤ 3 = fraud signal)
    3. Add uid1-level mean/std/max aggregations for all C cols

    Parameters
    ----------
    train_idx : indices of train rows within df. If None, uses all rows.
                ALWAYS pass train_idx when df contains val/test rows to avoid leakage.
    """
    eps = 1.0  # avoid divide-by-zero

    # ── Ratio features ────────────────────────────────────────────────────────
    if "C1" in df.columns and "C2" in df.columns:
        df["C1_C2_ratio"] = df["C1"] / (df["C2"] + eps)

    if "C5" in df.columns and "C6" in df.columns:
        df["C5_C6_ratio"] = df["C5"] / (df["C6"] + eps)

    if "C6" in df.columns and "C11" in df.columns:
        df["C6_C11_ratio"] = df["C6"] / (df["C11"] + eps)  # used in VAE

    # ── C3 fraud flag (low C3 = fraud signal) ─────────────────────────────────
    if "C3" in df.columns:
        df["C3_flag"] = (df["C3"] <= 3).astype(int)

    # ── UID aggregations (train-only fit to prevent leakage) ─────────────────
    source = df.iloc[train_idx] if train_idx is not None else df
    present_c = [c for c in C_COLS if c in df.columns]
    for c in present_c:
        grp = source.groupby("uid1")[c]
        df[f"{c}_uid_mean"] = df["uid1"].map(grp.mean()).fillna(-999)
        df[f"{c}_uid_std"]  = df["uid1"].map(grp.std()).fillna(-999)
        df[f"{c}_uid_max"]  = df["uid1"].map(grp.max()).fillna(-999)

    priority_added = [f"{c}_uid_mean" for c in PRIORITY_C_AGGS if c in df.columns]
    logger.info(
        "C features added: ratios, C3_flag, {} uid-agg columns.  "
        "Priority aggs: {}.",
        len(present_c) * 3, priority_added,
    )
    return df
