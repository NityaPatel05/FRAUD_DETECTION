from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from loguru import logger


M_COLS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]
TREE_NAN_FILL = -999


# ─────────────────────────────────────────────────────────
# M column encoding (must happen before any numeric fill)
# ─────────────────────────────────────────────────────────

def encode_m_columns(
    df: pd.DataFrame,
    nan_value: int = -1,
    m_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Encode M columns: T→1, F→0, NaN→nan_value.

    nan_value=-1  for tree models
    nan_value=0   for VAE (neutral, doesn't create spurious negative signal)
    """
    if m_cols is None:
        m_cols = [c for c in M_COLS if c in df.columns]

    for col in m_cols:
        df[col] = df[col].map({"T": 1, "F": 0}).fillna(nan_value).astype(int)

    logger.info(
        "Encoded {} M-columns (NaN → {}).  Sample: {}",
        len(m_cols), nan_value, dict(df[m_cols[:3]].iloc[0]) if len(df) > 0 else {},
    )
    return df


# ─────────────────────────────────────────────────────────
# Tree model NaN fill (-999)
# ─────────────────────────────────────────────────────────

def fill_nan_for_trees(
    df: pd.DataFrame,
    fill_value: int = TREE_NAN_FILL,
    exclude_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fill all remaining NaN values with -999 for XGBoost / LightGBM.

    M columns must already be encoded (encode_m_columns called first).
    String/object columns are left as-is (CatBoost handles them natively).

    Parameters
    ----------
    exclude_cols : columns to skip (e.g. CatBoost raw string cols, target)
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = [
        c for c, dtype in df.dtypes.items()
        if np.issubdtype(dtype, np.number) and c not in exclude_cols
    ]

    before_nan = sum(df[c].isna().sum() for c in numeric_cols)
    for col in numeric_cols:
        df[col] = df[col].fillna(fill_value)
    after_nan = sum(df[c].isna().sum() for c in numeric_cols)

    logger.info(
        "Tree NaN fill (-999): {} NaN values filled across {} numeric columns.",
        before_nan, len(numeric_cols),
    )
    assert after_nan == 0, "Remaining NaN after tree fill — check exclude_cols"
    return df


# ─────────────────────────────────────────────────────────
# DL model median imputation (after feature creation)
# ─────────────────────────────────────────────────────────

def fit_median_imputer(
    X_train: np.ndarray,
) -> SimpleImputer:
    """Fit median imputer on TRAIN ONLY.  Apply to val/test separately."""
    imp = SimpleImputer(strategy="median")
    imp.fit(X_train)
    logger.info("Median imputer fitted on {} rows × {} cols.", *X_train.shape)
    return imp


def apply_median_imputer(
    imp: SimpleImputer,
    X: np.ndarray,
    split_name: str = "data",
) -> np.ndarray:
    """Transform a split using a pre-fitted imputer."""
    X_imputed = imp.transform(X)
    remaining = np.isnan(X_imputed).sum()
    logger.info(
        "Median impute applied to {} (shape {}).  Remaining NaN: {}",
        split_name, X.shape, remaining,
    )
    return X_imputed


# ─────────────────────────────────────────────────────────
# CatBoost — leave string cols as-is, fill numeric only
# ─────────────────────────────────────────────────────────

def fill_nan_for_catboost(
    df: pd.DataFrame,
    cat_cols: List[str],
) -> pd.DataFrame:
    """
    For CatBoost:
      - Numeric cols → LEAVE AS NaN (CatBoost nan_mode handles this natively)
      - String cat_cols → fill 'missing' (CatBoost handles as a category)
      - M cols must be encoded BEFORE calling this
    """
    filled_cat = 0
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)
            filled_cat += 1

    logger.info(
        "CatBoost NaN fill: {} cat cols → 'missing'. Numeric NaNs left untouched.",
        filled_cat,
    )
    return df
