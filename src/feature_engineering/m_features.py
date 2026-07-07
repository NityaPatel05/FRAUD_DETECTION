from __future__ import annotations

from typing import List, Optional

import pandas as pd
import numpy as np
from loguru import logger


M_COLS = ["M1","M2","M3","M4","M5","M6","M7","M8","M9"]


def build_m_features(
    df: pd.DataFrame,
    m_cols: Optional[List[str]] = None,
    train_idx: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Build M column aggregation and mismatch features.

    Assumes M columns already encoded: T=1, F=0, NaN=-1 (trees) or NaN=0 (VAE).
    Calling encode_m_columns() in missing_values.py must precede this.

    Parameters
    ----------
    train_idx : indices of train rows within df. Stats are computed on train only.
                Pass None only when df is already an isolated train split.

    Adds:
      M_match_count, M_mismatch_count, M_nan_count
      M{n}_uid_mean for all M cols (M9_uid_mean, M5_uid_mean = top features)
    """
    if m_cols is None:
        m_cols = [c for c in M_COLS if c in df.columns]

    if not m_cols:
        logger.warning("No M columns found in df. Skipping M features.")
        return df

    # ── Match / mismatch / NaN count per row ──────────────────────────────────
    m_data = df[m_cols]

    df["M_match_count"]    = (m_data == 1).sum(axis=1)
    df["M_mismatch_count"] = (m_data == 0).sum(axis=1)
    df["M_nan_count"]      = (m_data == -1).sum(axis=1)

    # ── UID mean aggregations (M9 and M5 are top features) ────────────────────
    source = df.iloc[train_idx] if train_idx is not None else df
    for m in m_cols:
        grp = source.groupby("uid1")[m]
        df[f"{m}_uid_mean"] = df["uid1"].map(grp.mean()).fillna(-999)

    logger.info(
        "M features added: M_match_count, M_mismatch_count, M_nan_count, "
        "{} M_uid_mean cols.  "
        "Mean match: {:.2f}, mean mismatch: {:.2f}.",
        len(m_cols),
        df["M_match_count"].mean(),
        df["M_mismatch_count"].mean(),
    )
    return df
