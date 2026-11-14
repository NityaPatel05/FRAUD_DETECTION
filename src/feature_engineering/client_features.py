"""
Client-level (entity) behavioural features for the v2 tree pipeline.

General fraud practice: there is no customer id, so build a surrogate client key
and describe each transaction relative to that client's history. The key itself is
NEVER a model feature (most future clients are unseen) — it is only used to group.

Client key: uid1 = card1 _ addr1 _ (day - D1)   [D1 = days since card/account start]
            uid2 = uid1 _ P_emaildomain         [stricter]

``mode``:
  causal       — every aggregate uses only the client's rows up to the current one.
                 Servable in real time; this is the honest production estimate.
  transductive — aggregates over all rows (train+val+test), label-free. Matches
                 Kaggle 1st place; optimistic on a time holdout, not servable online.
Velocity, recency, lags and first-seen flags are causal in both modes.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from src.feature_engineering.entity_ops import (
    _Sorted, make_codes, entity_mean_std, entity_count, entity_nunique,
    first_time_pair, window_count_sum, prev_row_values,
)
from src.feature_engineering.uid_features import _int_str

C_FOR_MEAN = ["C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14"]
M_FOR_MEAN = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]
UID_MEAN_STD = ["TransactionAmt", "D4", "D9", "D10", "D15", "dist1"]
UID_NUNIQUE = [
    "P_emaildomain", "R_emaildomain", "dist1", "addr2", "cents", "ProductCD",
    "DeviceInfo", "device_fp", "id_02", "id_13", "id_17", "id_19", "id_20", "id_31",
    "C13", "V127", "V136", "V307", "V309", "V314", "V320",
]
COUNT_ENCODE = [
    "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1",
    "P_emaildomain", "R_emaildomain", "DeviceInfo", "device_fp",
    "id_30", "id_31", "id_33", "card1_addr1", "card1_addr1_P_emaildomain",
    "card2_dist1", "uid1", "uid2",
]
FIRST_SEEN_VALUES = ["P_emaildomain", "DeviceInfo", "device_fp", "id_31", "addr2", "ProductCD"]


def build_uid_keys(df: pd.DataFrame) -> pd.DataFrame:
    """uid1 / uid2 as strings (NaN when card1, addr1 or D1 is missing)."""
    parts_ok = df["card1"].notna() & df["addr1"].notna() & df["D1n"].notna()
    uid1 = _int_str(df["card1"]) + "_" + _int_str(df["addr1"]) + "_" + _int_str(df["D1n"])
    email_ok = parts_ok & df["P_emaildomain"].notna()
    uid2 = uid1 + "_" + df["P_emaildomain"].astype(str)
    df["uid1"] = uid1.where(parts_ok, other=None)
    df["uid2"] = uid2.where(email_ok, other=None)
    logger.info("uid1: {} unique ({:.1%} rows keyed) | uid2: {} unique",
                df["uid1"].nunique(), parts_ok.mean(), df["uid2"].nunique())
    return df


def _f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def add_count_encoding(df: pd.DataFrame, t: np.ndarray, mode: str, window_days: int,
                       cols: Sequence[str] = COUNT_ENCODE) -> Dict[str, np.ndarray]:
    """Popularity of a value.

    transductive: total count over all rows.
    causal      : count within the trailing ``window_days`` (stationary over time,
                  unlike an ever-growing cumulative count).
    """
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        codes = make_codes(df, c)
        if mode == "transductive":
            out[f"{c}_count"] = _f32(entity_count(codes, t, "transductive"))
        else:
            out[f"{c}_count{window_days}d"] = _f32(
                window_count_sum(codes, t, window_days * 86400)["count"])
    return out


def add_group_aggregates(df: pd.DataFrame, t: np.ndarray, mode: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    # ── by client (uid1) ─────────────────────────────────────────────────────
    codes = make_codes(df, "uid1")
    s = _Sorted(codes, t)
    for v in UID_MEAN_STD:
        if v in df.columns:
            st = entity_mean_std(codes, t, df[v].to_numpy(dtype=np.float64), mode, s=s)
            out[f"uid_{v}_mean"] = _f32(st["mean"])
            out[f"uid_{v}_std"] = _f32(st["std"])
    for v in C_FOR_MEAN + M_FOR_MEAN:
        if v in df.columns:
            st = entity_mean_std(codes, t, df[v].to_numpy(dtype=np.float64), mode, s=s)
            out[f"uid_{v}_mean"] = _f32(st["mean"])
    if "C14" in df.columns:
        out["uid_C14_std"] = _f32(entity_mean_std(codes, t, df["C14"].to_numpy(dtype=np.float64), mode, s=s)["std"])
    for v in UID_NUNIQUE:
        if v in df.columns:
            out[f"uid_{v}_nunique"] = _f32(entity_nunique(codes, t, make_codes(df, v), mode, s=s))
    out["uid_tx_count"] = _f32(entity_count(codes, t, mode, s=s))

    # ── by coarser keys (card BIN / card+address / +email) ──────────────────
    for g in ["card1", "card1_addr1", "card1_addr1_P_emaildomain"]:
        gcodes = make_codes(df, g)
        gs = _Sorted(gcodes, t)
        for v in ["TransactionAmt", "D9", "D11"]:
            if v in df.columns:
                st = entity_mean_std(gcodes, t, df[v].to_numpy(dtype=np.float64), mode, s=gs)
                out[f"{g}_{v}_mean"] = _f32(st["mean"])
                out[f"{g}_{v}_std"] = _f32(st["std"])
    logger.info("Group aggregates ({}): {} columns.", mode, len(out))
    return out


def add_behaviour_features(df: pd.DataFrame, t: np.ndarray,
                           windows_sec: Sequence[int]) -> Dict[str, np.ndarray]:
    """Velocity, deviation from own history, recency, lags, novelty — always causal."""
    out: Dict[str, np.ndarray] = {}
    codes = make_codes(df, "uid1")
    s = _Sorted(codes, t)
    amt = df["TransactionAmt"].to_numpy(dtype=np.float64)

    # Velocity: count & amount over sliding windows
    for w in windows_sec:
        tag = f"{w // 3600}h" if w < 86400 else f"{w // 86400}d"
        r = window_count_sum(codes, t, w, values=amt, s=s)
        out[f"uid_n_{tag}"] = _f32(r["count"])
        out[f"uid_amt_sum_{tag}"] = _f32(r["sum"])

    # Deviation from the client's OWN past (strictly prior rows)
    prior = entity_mean_std(codes, t, amt, "causal", include_current=False, s=s)
    std_floor = np.maximum(np.nan_to_num(prior["std"], nan=0.0), 1.0)
    z = (amt - prior["mean"]) / std_floor
    out["amt_z_vs_own_past"] = _f32(np.clip(z, -10, 10))           # NaN if no history
    out["amt_ratio_vs_own_past"] = _f32(amt / np.maximum(prior["mean"], 0.01))

    # Recency / lag features (cheap replacement for a sequence model)
    prev_t = prev_row_values(codes, t, t.astype(np.float64), s=s)
    out["uid_hours_since_prev"] = _f32((t - prev_t) / 3600.0)
    prev_amt = prev_row_values(codes, t, amt, s=s)
    out["uid_prev_amt"] = _f32(prev_amt)
    out["uid_amt_diff_prev"] = _f32(amt - prev_amt)
    if "ProductCD" in df.columns:
        pc = make_codes(df, "ProductCD").astype(np.float64)
        prev_pc = prev_row_values(codes, t, pc, s=s)
        out["uid_same_product_as_prev"] = _f32(np.where(np.isnan(prev_pc), np.nan, (pc == prev_pc).astype(float)))
    first_t = s.to_orig(s.t[s.start])  # rows are time-sorted inside each client block
    out["uid_days_since_first"] = _f32((t - first_t) / 86400.0)
    out["uid_tx_index"] = _f32(entity_count(codes, t, "causal", s=s) - 1)

    # Novelty: first time this client uses this email / device / browser …
    for v in FIRST_SEEN_VALUES:
        if v in df.columns:
            out[f"uid_new_{v}"] = _f32(first_time_pair(codes, t, make_codes(df, v), s=s))
    logger.info("Behaviour features: {} columns.", len(out))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3 — behaviour profiles (replaces the BiLSTM sequence model)
# Production practice: Feedzai profiles + decayed aggregates, Featurespace
# behavioural analytics, Sift velocity, Coinbase last-N sequence features.
# ═════════════════════════════════════════════════════════════════════════════

def add_behaviour_v3(df: pd.DataFrame, t: np.ndarray, half_lives_sec: Sequence[int]) -> Dict[str, np.ndarray]:
    from src.feature_engineering.streaming_ops import decayed_profile, lag_values

    out: Dict[str, np.ndarray] = {}
    amt = df["TransactionAmt"].to_numpy(dtype=np.float64)
    uid = make_codes(df, "uid1")
    s = _Sorted(uid, t)

    # 3a. velocity on more entities + short window + ratio features
    r10m = window_count_sum(uid, t, 600, values=amt, s=s)
    out["uid_n_10m"] = _f32(r10m["count"])
    r1h = window_count_sum(uid, t, 3600, s=s)["count"]
    r1d = window_count_sum(uid, t, 86400, values=amt, s=s)
    r2d = window_count_sum(uid, t, 2 * 86400, values=amt, s=s)
    r7d = window_count_sum(uid, t, 7 * 86400, s=s)["count"]
    out["uid_n1h_vs_7d_rate"] = _f32(r1h / (r7d / 168.0))                   # burst vs own weekly pace
    prev_day_sum = r2d["sum"] - r1d["sum"]
    out["uid_amt_1d_vs_prev_1d"] = _f32(r1d["sum"] / np.maximum(prev_day_sum, 1.0))
    for ent in ["card1_addr1", "device_fp"]:
        if ent in df.columns:
            ec = make_codes(df, ent)
            es = _Sorted(ec, t)
            for w, tag in [(3600, "1h"), (86400, "1d"), (7 * 86400, "7d")]:
                r = window_count_sum(ec, t, w, values=amt, s=es)
                out[f"{ent}_n_{tag}"] = _f32(r["count"])
                out[f"{ent}_amt_sum_{tag}"] = _f32(r["sum"])

    # 3b. exponentially decayed profiles of the client (prior rows only)
    hl_tag = {3600: "1h", 86400: "1d", 7 * 86400: "7d", 30 * 86400: "30d"}
    counts = {}
    for hl in half_lives_sec:
        tag = hl_tag.get(int(hl), f"{int(hl)}s")
        p = decayed_profile(uid, t, amt, hl)
        counts[hl] = p["count"]
        out[f"uid_ema_n_{tag}"] = _f32(p["count"])
        out[f"uid_ema_amt_mean_{tag}"] = _f32(p["mean"])
        z = (amt - p["mean"]) / np.maximum(np.nan_to_num(p["std"], nan=0.0), 1.0)
        out[f"uid_ema_amt_z_{tag}"] = _f32(np.clip(z, -10, 10))
    hls = sorted(counts)
    if len(hls) >= 2:   # "delayed EMA": short-term activity vs long-term pace
        short, long_ = hls[0], hls[-1]
        out["uid_ema_burst"] = _f32(counts[short] / (counts[long_] * short / long_ + 1e-3))

    # 3c. habit deviation: how usual is this hour for the client (prior rows)
    if "hour" in df.columns:
        # (client, hour) key without copying the frame
        hour_codes = np.where(uid >= 0, uid * 24 + df["hour"].to_numpy().astype(np.int64), -1)
        n_same_hour_prior = entity_count(hour_codes, t, "causal") - 1
        n_prior = s.to_orig(s.rank.astype(np.float64))
        with np.errstate(invalid="ignore", divide="ignore"):
            out["uid_hour_typicality"] = _f32(np.where(n_prior > 0, n_same_hour_prior / n_prior, np.nan))
    out["uid_history_len"] = _f32(s.to_orig(s.rank.astype(np.float64)))

    # 3d. last-k lags (k = 2, 3) and "changed since last transaction" flags
    for k in (2, 3):
        out[f"uid_amt_lag{k}"] = _f32(lag_values(uid, t, amt, k))
        lag_t = lag_values(uid, t, t.astype(np.float64), k)
        out[f"uid_hours_since_lag{k}"] = _f32((t - lag_t) / 3600.0)
    for col in ["device_fp", "DeviceInfo", "P_emaildomain", "R_emaildomain", "id_31", "id_30", "dist1"]:
        if col in df.columns:
            vc = make_codes(df, col).astype(np.float64)
            vc[vc < 0] = np.nan
            prev = prev_row_values(uid, t, vc, s=s)
            out[f"uid_changed_{col}"] = _f32(np.where(np.isnan(prev) | np.isnan(vc), np.nan, (vc != prev).astype(float)))
    logger.info("Behaviour v3 features: {} columns.", len(out))
    return out
