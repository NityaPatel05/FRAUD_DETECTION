from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


FREQ_COLS = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "DeviceInfo", "DeviceType",
    "uid1", "uid2", "uid_magic",
    "id_20", "id_31",
]


# ─────────────────────────────────────────────────────────
# Fit: compute frequency map on train split only
# ─────────────────────────────────────────────────────────

def fit_freq_encoder(
    train_df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    train_idx: Optional[np.ndarray] = None,
) -> Dict[str, Dict]:
    """
    Compute frequency (value_counts normalised) for each column.
    If train_idx is provided, compute on those rows only.

    Returns {col: {value: frequency, ...}}
    """
    if cols is None:
        cols = FREQ_COLS

    if train_idx is not None:
        source = train_df.iloc[train_idx]
    else:
        source = train_df

    freq_maps: Dict[str, Dict] = {}
    for col in cols:
        if col not in source.columns:
            logger.warning("Column '{}' not found — skipping freq encode.", col)
            continue
        freq_map = source[col].value_counts(normalize=True).to_dict()
        freq_maps[col] = freq_map
        logger.debug(
            "Freq map for '{}': {} unique values, top-5 freqs: {}",
            col, len(freq_map),
            sorted(freq_map.values(), reverse=True)[:5],
        )

    logger.info(
        "Frequency encoder fitted on {} rows for {} columns.",
        len(source), len(freq_maps),
    )
    return freq_maps


# ─────────────────────────────────────────────────────────
# Apply: map frequency to a dataframe split
# ─────────────────────────────────────────────────────────

def apply_freq_encoder(
    df: pd.DataFrame,
    freq_maps: Dict[str, Dict],
    fillna_val: float = 0.0,
) -> pd.DataFrame:
    """
    Add <col>_freq columns to df.
    Unseen values (in val/test but not in train) are filled with 0.
    """
    for col, freq_map in freq_maps.items():
        if col not in df.columns:
            continue
        df[f"{col}_freq"] = df[col].map(freq_map).fillna(fillna_val)

    logger.info("Applied frequency encoding: added {} _freq columns.", len(freq_maps))
    return df


# ─────────────────────────────────────────────────────────
# Label encoding for low-cardinality columns
# ─────────────────────────────────────────────────────────

def fit_label_encoder(
    train_df: pd.DataFrame,
    cols: List[str],
) -> Dict[str, Dict]:
    """
    Fit label-encoding maps (str → int) from train data.
    Unseen val/test values will be mapped to -1.
    """
    label_maps: Dict[str, Dict] = {}
    for col in cols:
        if col not in train_df.columns:
            continue
        unique_vals = train_df[col].astype(str).unique()
        label_maps[col] = {v: i for i, v in enumerate(sorted(unique_vals))}
    logger.info("Label encoder fitted for {} columns.", len(label_maps))
    return label_maps


def apply_label_encoder(
    df: pd.DataFrame,
    label_maps: Dict[str, Dict],
    unknown_val: int = -1,
) -> pd.DataFrame:
    """Apply label encoders.  Unseen values → unknown_val."""
    for col, lmap in label_maps.items():
        if col not in df.columns:
            continue
        df[f"{col}_encoded"] = df[col].astype(str).map(lmap).fillna(unknown_val).astype(int)
    logger.info("Applied label encoding: {} columns.", len(label_maps))
    return df
