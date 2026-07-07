from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────
# Generic helper: fit aggregation stats on train
# ─────────────────────────────────────────────────────────

def fit_uid_aggs(
    train_df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
    agg_funcs: List[str] = ("mean", "std", "max", "min", "count"),
) -> Dict[str, pd.Series]:
    """
    Compute group statistics on TRAIN split.
    Returns {f"{group_col}_{value_col}_{func}": pd.Series(indexed by group_col value)}
    """
    stats: Dict[str, pd.Series] = {}
    for gcol in group_cols:
        if gcol not in train_df.columns:
            continue
        grp = train_df.groupby(gcol)[value_col]
        for func in agg_funcs:
            key = f"{gcol}_{value_col}_{func}"
            stats[key] = grp.agg(func)
    return stats


def apply_uid_aggs(
    df: pd.DataFrame,
    stats: Dict[str, pd.Series],
    join_col_map: Dict[str, str],  # {stat_key: group_col_used_to_join}
    fill_val: float = -999,
) -> pd.DataFrame:
    """
    Map pre-computed stats back onto df rows via merge-style map.
    Unseen values → fill_val.
    """
    for key, series in stats.items():
        gcol = join_col_map[key]
        if gcol not in df.columns:
            continue
        df[key] = df[gcol].map(series).fillna(fill_val)
    return df


# ─────────────────────────────────────────────────────────
# Main: build all UID aggregation features
# ─────────────────────────────────────────────────────────

def build_uid_aggregations(
    df: pd.DataFrame,
    train_idx: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Build all UID / card1 / addr1 / email aggregation features.

    If train_idx is provided, statistics are computed on those rows only
    and then mapped to the entire dataframe (time-safe).
    Otherwise, compute on the full df (use only when df IS the training set).

    Adds:
      TransactionAmt_{uid/card1/addr1/email}_{mean/std/max/min/count}
      C{1,2,3,4,5,6,7,8,9,11,13,14}_uid_{mean/std/max}
      M{1-9}_uid_mean  (M9 and M5 explicitly named in top-5% solution)
      D{1,2,3,4,5,10,11,15}_norm_uid_{mean/std}
      V top-8_uid_{mean/std}  (handled in v_column_reduction.py)
    """
    source = df.iloc[train_idx] if train_idx is not None else df

    agg_funcs = ["mean", "std", "max", "min", "count"]

    # ── TransactionAmt aggregations ──────────────────────────────────────────
    for group_col in ["uid1", "uid_magic", "card1", "addr1", "P_emaildomain"]:
        if group_col not in df.columns:
            continue
        grp = source.groupby(group_col)["TransactionAmt"]
        stats = {f: grp.agg(f) for f in agg_funcs}

        for f, series in stats.items():
            col_name = f"{group_col.replace('P_emaildomain','email')}_amt_{f}" \
                       if group_col == "P_emaildomain" else f"{group_col}_amt_{f}"
            # For count, rename to tx_count
            if f == "count":
                col_name = col_name.replace("_amt_count", "_tx_count")
            df[col_name] = df[group_col].map(series).fillna(-999)

    # ── C column uid aggregations (explicitly named in top-5% solution) ───────
    c_cols = [c for c in ["C1","C2","C3","C4","C5","C6","C7","C8","C9","C11","C13","C14"]
              if c in df.columns]
    for c in c_cols:
        grp = source.groupby("uid1")[c]
        for f in ["mean", "std", "max"]:
            df[f"{c}_uid_{f}"] = df["uid1"].map(grp.agg(f)).fillna(-999)

    # ── M column uid aggregations (M9_uid_mean, M5_uid_mean = top features) ──
    m_cols = [m for m in ["M1","M2","M3","M4","M5","M6","M7","M8","M9"]
              if m in df.columns]
    for m in m_cols:
        grp = source.groupby("uid1")[m]
        df[f"{m}_uid_mean"] = df["uid1"].map(grp.mean()).fillna(-999)

    # ── Normalized D column uid aggregations ─────────────────────────────────
    d_norm_cols = [c for c in df.columns
                   if c.startswith("D") and c.endswith("_norm")]
    priority_d = ["D1_norm","D2_norm","D3_norm","D4_norm","D5_norm",
                  "D10_norm","D11_norm","D15_norm"]
    d_to_agg = [c for c in priority_d if c in df.columns]

    for d in d_to_agg:
        grp = source.groupby("uid1")[d]
        df[f"{d}_uid_mean"] = df["uid1"].map(grp.mean()).fillna(-999)
        df[f"{d}_uid_std"]  = df["uid1"].map(grp.std()).fillna(-999)

    # ── Log newly added columns ───────────────────────────────────────────────
    new_cols = [c for c in df.columns if "_uid_" in c or "_amt_" in c or "_tx_count" in c]
    logger.info("UID aggregation features added: {} total.", len(new_cols))
    return df
