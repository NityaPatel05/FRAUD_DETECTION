"""
v2 feature pipeline for the tree models (LightGBM / XGBoost / CatBoost).

Design (each point fixes a bug found in the audit):
  1. Takes train / val / test ALREADY split — no second time_split inside.
  2. Builds features ONCE on the time-sorted concat of all rows, so a client's
     history flows from train into val into test (no train-only stats, no -999
     cliff for the 58% of val rows whose client is new).
  3. No adversarial "drift" filter. Feature selection is done later with a
     time-consistency check (scripts/run_feature_selection.py).
  4. NaN stays NaN (GBDTs handle it natively) — no -999 sentinel that later
     gets divided by or averaged in.
  5. Every string column is label-encoded instead of silently dropped.
  6. ``agg_mode`` = causal (servable, honest) | transductive (Kaggle-style).
  7. Delayed label-history + graph statistics replace the VAE / GNN models.
"""
from __future__ import annotations

import gc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.feature_engineering.base_features import (
    build_base_features, label_encode_strings, NON_FEATURES,
)
from src.feature_engineering.client_features import (
    build_uid_keys, add_count_encoding, add_group_aggregates, add_behaviour_features,
    add_behaviour_v3,
)
from src.feature_engineering.derived_signals import (
    DEFAULT_DROP, vendor_engineered_cols, add_derived_signals,
)
from src.feature_engineering.entity_ops import make_codes
from src.feature_engineering.graph_features import add_graph_features, add_network_v4
from src.feature_engineering.label_history import add_label_history
from src.preprocessing.v_column_reduction import group_v_cols_by_nan, reduce_v_keep_most_unique

# High-cardinality key strings: used for grouping/counting, never as raw features
# (the model must generalise to clients it has never seen).
KEY_COLS = ["uid1", "uid2", "card1_addr1", "card1_addr1_P_emaildomain", "card2_dist1", "device_fp"]

# Columns treated as categorical by CatBoost (they are integer codes in the frame).
CATBOOST_CAT_COLS = [
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "P_email_prefix", "R_email_prefix",
    "DeviceType", "DeviceInfo", "device_brand", "device_os", "browser",
    "id_30", "id_31", "id_33", "M4",
    "browser_family", "os_family", "P_email_class", "R_email_class", "proxy_type",
]

DEFAULTS = {
    "agg_mode": "causal",
    "label_delay_days": 30,
    "label_smoothing": 20.0,
    "velocity_windows_sec": [3600, 86400, 604800, 2592000],
    "count_window_days": 30,
    "v_corr_threshold": 0.75,
    "use_label_history": True,
    "use_graph_features": True,
    # schema: "full" = every IEEE-CIS column; "raw" = only fields a real payment
    # gateway receives (drops Vesta's pre-computed V/C/D2-15/id_01-11 columns)
    "schema": "full",
    "drop_cols": None,             # None → derived_signals.DEFAULT_DROP
    "use_derived_signals": True,   # Phase 2
    "use_behaviour_v3": True,      # Phase 3 (replaces BiLSTM)
    "use_network_v4": True,        # Phase 4 (replaces GNN)
    "use_hmm_features": True,      # Phase 3e (Lucas et al. HMM sequence surprise)
    "hmm_states": 5,
    "ema_half_lives_sec": [3600, 86400, 604800, 2592000],
    "hub_cap": 20,
}


def dataset_tag(mode: str, schema: str = "full") -> str:
    """Folder suffix for a feature set: causal | transductive | causal_raw | …"""
    return mode if schema == "full" else f"{mode}_{schema}"


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    f64 = df.select_dtypes(include=["float64"]).columns
    keep = {"TransactionAmt"}  # precise cents already extracted
    df[[c for c in f64 if c not in keep]] = df[[c for c in f64 if c not in keep]].astype(np.float32)
    return df


def run_feature_pipeline_v2(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fe_cfg: Dict | None = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    cfg = {**DEFAULTS, **(fe_cfg or {})}
    mode = cfg["agg_mode"]
    if mode not in ("causal", "transductive"):
        raise ValueError(f"agg_mode must be causal|transductive, got {mode}")
    logger.info("=== v2 feature pipeline | agg_mode={} ===", mode)

    # ── 0. one time-ordered frame ────────────────────────────────────────────
    parts = []
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        # float32 per part BEFORE concatenating keeps peak RAM low (~1.1M rows x 430 cols)
        part = _downcast(part.copy())
        part.columns = [c.replace("id-", "id_") if c.startswith("id-") else c for c in part.columns]
        part["split"] = name
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()                                   # release the per-split copies
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    if "isFraud" not in df.columns:
        df["isFraud"] = np.nan
    df.loc[df["split"] == "test", "isFraud"] = np.nan
    logger.info("Rows: {}", df["split"].value_counts().to_dict())

    # ── 0b. column removal: data-study drop list + optional real-world schema ──
    drop = list(DEFAULT_DROP if cfg["drop_cols"] is None else cfg["drop_cols"])
    if cfg["schema"] == "raw":
        drop += vendor_engineered_cols(df.columns)
    elif cfg["schema"] != "full":
        raise ValueError(f"schema must be full|raw, got {cfg['schema']}")
    drop = [c for c in dict.fromkeys(drop) if c in df.columns]
    df = df.drop(columns=drop)
    logger.info("schema={} | removed {} columns", cfg["schema"], len(drop))
    # raw input contract for the online engine (columns + dtypes it must reproduce)
    raw_columns = [c for c in df.columns if c not in ("split", "isFraud")]
    raw_dtypes = {c: str(df[c].dtype) for c in raw_columns}

    # ── 1. row-level features ────────────────────────────────────────────────
    build_base_features(df)
    build_uid_keys(df)
    df = _downcast(df)
    t = df["TransactionDT"].to_numpy(dtype=np.int64)
    is_train = (df["split"] == "train").to_numpy()

    # ── 2. V-column reduction (fit on train rows, label-free) ────────────────
    v_groups = group_v_cols_by_nan(df.loc[is_train])
    kept_v = reduce_v_keep_most_unique(df.loc[is_train], v_groups, cfg["v_corr_threshold"])
    all_v = [c for g in v_groups.values() for c in g]
    df = df.drop(columns=[c for c in all_v if c not in kept_v])

    # ── 3. entity features ──────────────────────────────────────────────────
    base_cols = [c for c in df.columns]
    y = df["isFraud"].to_numpy(dtype=np.float64)
    prior = float(np.nanmean(y[is_train]))
    families: Dict[str, List[str]] = {}
    feats: Dict[str, np.ndarray] = {}
    hmm_params: Dict = {}

    def add(family: str, block: Dict) -> None:
        families[family] = list(block)
        feats.update(block)

    if cfg["use_derived_signals"]:
        add("derived_signals", add_derived_signals(df))
    add("count_encoding", add_count_encoding(df, t, mode, cfg["count_window_days"]))
    add("client_aggregates", add_group_aggregates(df, t, mode))
    add("behaviour", add_behaviour_features(df, t, cfg["velocity_windows_sec"]))
    if cfg["use_behaviour_v3"]:
        add("behaviour_v3", add_behaviour_v3(df, t, cfg["ema_half_lives_sec"]))
    if cfg["use_hmm_features"]:
        from src.feature_engineering.hmm_features import add_hmm_features
        hmm_params: Dict = {}
        add("hmm_sequence", add_hmm_features(df, t, is_train, cfg["hmm_states"], params_out=hmm_params))
    if cfg["use_graph_features"]:
        add("graph", add_graph_features(df, t, mode))
    if cfg["use_network_v4"]:
        add("network_v4", add_network_v4(df, t, y, cfg["label_delay_days"], cfg["label_smoothing"],
                                         prior, cfg["hub_cap"]))
    if cfg["use_label_history"]:
        add("label_history", add_label_history(df, t, y, cfg["label_delay_days"],
                                               cfg["label_smoothing"], prior))
    df = pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)
    del feats
    gc.collect()                                   # release the feature arrays (now inside df)

    # ── 4. encode remaining strings; assemble feature list ──────────────────
    # integer client id kept as a NON-feature column (folds, post-processing, CP@k)
    df["uid1_code"] = make_codes(df, "uid1")
    df = df.drop(columns=[c for c in KEY_COLS if c in df.columns])
    df, label_maps = label_encode_strings(df, exclude=["split"])
    feature_cols: List[str] = [c for c in df.columns if c not in NON_FEATURES and c != "uid1_code"]
    constant = [c for c in feature_cols if df.loc[is_train, c].nunique(dropna=False) <= 1]
    feature_cols = [c for c in feature_cols if c not in constant]
    if constant:
        logger.info("Dropped {} constant columns: {}", len(constant), constant[:10])

    families["raw_columns"] = [c for c in base_cols if c in feature_cols]
    families = {k: [c for c in v if c in feature_cols] for k, v in families.items()}
    meta = {
        "agg_mode": mode,
        "schema": cfg["schema"],
        "feature_families": families,
        "dropped_cols": drop,
        "raw_columns": raw_columns,
        "raw_dtypes": raw_dtypes,
        "dropped_v_cols": [c for c in all_v if c not in kept_v],
        "hmm_params": hmm_params if cfg["use_hmm_features"] else {},
        "fe_cfg": cfg,
        "feature_cols": feature_cols,
        "catboost_cat_cols": [c for c in CATBOOST_CAT_COLS if c in feature_cols],
        "kept_v_cols": kept_v,
        "label_maps": label_maps,
        "train_fraud_rate": float(df.loc[is_train, "isFraud"].mean()),
    }
    keep_cols = ["TransactionID", "TransactionDT", "day", "isFraud", "split", "uid1_code"] + feature_cols
    out = {name: df.loc[df["split"] == name, keep_cols].reset_index(drop=True)
           for name in ["train", "val", "test"]}
    logger.info("v2 features: {} | train {} val {} test {}", len(feature_cols),
                len(out["train"]), len(out["val"]), len(out["test"]))
    return out, meta
