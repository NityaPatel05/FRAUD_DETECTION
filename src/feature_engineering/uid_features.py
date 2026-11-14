from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


def _int_str(s: pd.Series) -> pd.Series:
    """Format a numeric key column as an integer string ('315', not '315.0').

    Batch and real-time paths must produce byte-identical keys, otherwise online
    lookups never match offline-computed client histories. NaN → 'nan'.
    """
    num = pd.to_numeric(s, errors="coerce")
    out = num.round().astype("Int64").astype(str)
    return out.replace("<NA>", "nan")


def _int_str_scalar(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "nan"
    return "nan" if np.isnan(f) else str(int(round(f)))


def build_uid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct all UID columns. Requires 'day' column (from sort_and_split.sort_by_time).

    Adds columns: uid1, uid2, uid_magic, uid3 (optional).
    """
    # ── uid1: card + address (primary UID) ───────────────────────────────────
    # Fix: D1 normalises client start-day only when subtracted from the current day
    d1n = (df["D1"] - df["day"]).round()
    df["uid1"] = (
        _int_str(df["card1"]) + "_" + _int_str(df["addr1"]) + "_" + _int_str(d1n)
    )

    # ── uid2: uid1 + email (stricter — same client + email combo) ────────────
    df["uid2"] = df["uid1"] + "_" + df["P_emaildomain"].astype(str)

    # ── uid_magic: kept as an alias of uid1 for backward compatibility ───────
    # (it was built with the identical formula; the duplicate added no signal)
    df["uid_magic"] = df["uid1"]

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
    d1   = row.get("D1", np.nan)
    d1   = np.nan if d1 is None else float(d1)
    d1_norm = _int_str_scalar(d1 - day) if not np.isnan(d1) else "nan"

    card1 = _int_str_scalar(row.get("card1", np.nan))
    addr1 = _int_str_scalar(row.get("addr1", np.nan))

    uid1     = f"{card1}_{addr1}_{d1_norm}"
    uid2     = f"{uid1}_{row.get('P_emaildomain', 'nan')}"
    uid_magic= uid1
    uid3     = f"{uid1}_{row.get('DeviceInfo', 'nan')}"

    return {
        "uid1": uid1,
        "uid2": uid2,
        "uid_magic": uid_magic,
        "uid3": uid3,
    }
