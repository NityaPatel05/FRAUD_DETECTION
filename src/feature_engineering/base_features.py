"""
Row-level (no history) features for the v2 tree pipeline.

Fixes vs. the legacy pipeline:
  * every string column reaches the model (label-encoded) instead of being silently
    dropped by an ``dtype != object`` filter (ProductCD, card4/6, emails, id_12..id_38…)
  * M4 (values M0/M1/M2) is encoded correctly
  * only "anchor-date" D columns are normalised (D - day); gap-type D columns stay raw
  * absolute-time columns (TransactionDT, day, week, month) are never features
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# D columns that count "days since <fixed past event>": D - day is ~constant per client.
# Gap-type columns (D1 is used inside the uid instead; D2, D3, D5, D9) stay raw.
D_NORMALISE = ["D4", "D6", "D7", "D8", "D10", "D11", "D12", "D13", "D14", "D15"]

M_TF_COLS = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]

# Columns that identify time or rows — never model features.
NON_FEATURES = {
    "TransactionID", "TransactionDT", "isFraud", "split",
    "day", "week", "month", "D1n",
}


def rename_identity_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Kaggle test uses id-01; train uses id_01."""
    df.columns = [c.replace("id-", "id_") if c.startswith("id-") else c for c in df.columns]
    return df


def encode_m(df: pd.DataFrame) -> pd.DataFrame:
    for c in M_TF_COLS:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].map({"T": 1.0, "F": 0.0})
    if "M4" in df.columns and df["M4"].dtype == object:
        df["M4"] = df["M4"].map({"M0": 0.0, "M1": 1.0, "M2": 2.0})
    return df


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """``day`` / ``hour`` / ``weekday``. ``day`` is for folds & uid only, not a feature."""
    df["day"] = (df["TransactionDT"] // 86400).astype(np.int32)
    df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
    df["weekday"] = (df["day"] % 7).astype(np.int8)
    return df


def normalise_d(df: pd.DataFrame) -> pd.DataFrame:
    """Dn = day - Dn for anchor-type D columns (replaces the raw column).

    Raw Dn grows with time for the same client (train/test drift); day - Dn is
    the (constant) date of the anchoring event.
    """
    for c in D_NORMALISE:
        if c in df.columns:
            df[c] = (df["day"] - df[c]).astype(np.float32)
    df["D1n"] = (df["day"] - df["D1"]).round()  # used for the uid only
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    amt = df["TransactionAmt"].astype(np.float64)
    df["TransactionAmt_log"] = np.log1p(amt).astype(np.float32)
    df["cents"] = (amt - np.floor(amt)).round(3).astype(np.float32)
    return df


def add_string_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap parsed versions of a few messy string columns."""
    for side in ["P", "R"]:
        col = f"{side}_emaildomain"
        if col in df.columns:
            df[f"{side}_email_prefix"] = df[col].astype("string").str.split(".").str[0]
    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        both = df["P_emaildomain"].notna() & df["R_emaildomain"].notna()
        df["P_eq_R_email"] = np.where(both, (df["P_emaildomain"] == df["R_emaildomain"]).astype(float), np.nan)
    if "id_30" in df.columns:
        df["device_os"] = df["id_30"].astype("string").str.split(" ").str[0]
    if "id_31" in df.columns:
        df["browser"] = df["id_31"].astype("string").str.split(" ").str[0]
    if "DeviceInfo" in df.columns:
        df["device_brand"] = df["DeviceInfo"].astype("string").str.split(r"[ /\-_]").str[0].str.lower()
    return df


def add_combo_keys(df: pd.DataFrame) -> pd.DataFrame:
    """String combo keys used for grouping/count encoding (not fed raw to models)."""
    def cat(*cols):
        out = df[cols[0]].astype("string")
        for c in cols[1:]:
            out = out + "_" + df[c].astype("string")
        return out  # NaN in any part → <NA>
    df["card1_addr1"] = cat("card1", "addr1")
    df["card1_addr1_P_emaildomain"] = cat("card1", "addr1", "P_emaildomain")
    df["card2_dist1"] = cat("card2", "dist1") if "dist1" in df.columns else pd.NA
    # Pseudo device fingerprint (data study): DeviceInfo + id_19/20 + browser + screen.
    # Defined only when DeviceInfo is known (no giant "unknown device" hub node).
    if "DeviceInfo" in df.columns:
        fp = df["DeviceInfo"].astype("string")
        for c in ["id_19", "id_20", "id_31", "id_33"]:
            if c in df.columns:
                fp = fp + "_" + df[c].astype("string").fillna("na")
        df["device_fp"] = fp.where(df["DeviceInfo"].notna(), other=pd.NA)
    else:
        df["device_fp"] = pd.NA
    return df


def label_encode_strings(df: pd.DataFrame, exclude: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Integer-encode every remaining string column (NaN stays NaN).

    The value→code map is label-free; codes are arbitrary ids, so building them
    on all rows leaks nothing. Unseen values at inference time map to NaN.
    Returns the encoded frame and {col: categories} for inference.
    """
    maps: Dict[str, List[str]] = {}
    str_cols = [c for c in df.columns
                if c not in exclude and (df[c].dtype == object or str(df[c].dtype) == "string")]
    for c in str_cols:
        codes, cats = pd.factorize(df[c].astype("string"), sort=True, use_na_sentinel=True)
        df[c] = np.where(codes < 0, np.nan, codes).astype(np.float32)
        maps[c] = list(cats)
    logger.info("Label-encoded {} string columns (e.g. {}).", len(str_cols), str_cols[:8])
    return df, maps


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    rename_identity_cols(df)
    encode_m(df)
    add_time_columns(df)
    normalise_d(df)
    add_amount_features(df)
    add_string_derivatives(df)
    add_combo_keys(df)
    return df
