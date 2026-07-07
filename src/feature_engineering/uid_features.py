from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


def build_uid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct all UID columns. Requires 'day' column (from sort_and_split.sort_by_time).

    Adds columns: uid1, uid2, uid_magic, uid3 (optional).
    """
    # ── uid1: card + address (primary UID) ───────────────────────────────────
    # Fix: D1 normalises client start-day only when subtracted from the current day
    df["uid1"] = (
        df["card1"].astype(str) + "_"
        + df["addr1"].astype(str) + "_"
        + (df["D1"] - df["day"]).round().astype(str)
    )

    # ── uid2: uid1 + email (stricter — same client + email combo) ────────────
    df["uid2"] = (
        df["uid1"].astype(str) + "_"
        + df["P_emaildomain"].astype(str)
    )

    # ── uid_magic: the 1st-place Deotte construction ─────────────────────────
    # D1_norm = D1 - day  (roughly constant per client regardless of when txn happened)
    # Use round() to snap noisy floats to integers
    df["uid_magic"] = (
        df["card1"].astype(str) + "_"
        + df["addr1"].astype(str) + "_"
        + (df["D1"] - df["day"]).round().astype(str)
    )

    # ── uid3: device ─ sparse, use only for additional features not as PK ────
    df["uid3"] = (
        df["uid1"].astype(str) + "_"
        + df["DeviceInfo"].astype(str)
    )

    # ── Diagnostics ──────────────────────────────────────────────────────────
    n = len(df)
    for uid in ["uid1", "uid2", "uid_magic", "uid3"]:
        uniq = df[uid].nunique()
        logger.info("{}:  {} unique values ({:.1f}% of rows)", uid, uniq, 100 * uniq / n)

    return df


def build_uid_features_inference(row: dict) -> dict:
    """
    Same logic as build_uid_features but for a single transaction dict.
    Used in real-time inference path.
    """
    day  = int(row.get("TransactionDT", 0)) // 86400
    d1   = float(row.get("D1", np.nan))
    d1_norm = round(d1 - day) if not np.isnan(d1) else "nan"
    
    card1 = row.get("card1", "nan")
    addr1 = row.get("addr1", "nan")

    uid1     = f"{card1}_{addr1}_{d1_norm}"
    uid2     = f"{uid1}_{row.get('P_emaildomain', 'nan')}"
    uid_magic= f"{card1}_{addr1}_{d1_norm}"
    uid3     = f"{uid1}_{row.get('DeviceInfo', 'nan')}"

    return {
        "uid1": uid1,
        "uid2": uid2,
        "uid_magic": uid_magic,
        "uid3": uid3,
    }
