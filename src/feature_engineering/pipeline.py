from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# Preprocessing
from src.preprocessing.sort_and_split import sort_by_time, time_split
from src.preprocessing.missing_values import (
    encode_m_columns, fill_nan_for_trees,
    fill_nan_for_catboost, fit_median_imputer, apply_median_imputer,
)
from src.preprocessing.v_column_reduction import (
    group_v_cols_by_nan, reduce_v_by_correlation,
    fit_v_pca, apply_v_pca, add_uid_v_aggregations,
)

# Feature engineering modules
from src.feature_engineering.uid_features       import build_uid_features
from src.feature_engineering.frequency_encoding import fit_freq_encoder, apply_freq_encoder
from src.feature_engineering.uid_aggregations   import build_uid_aggregations
from src.feature_engineering.amount_features    import build_amount_features
from src.feature_engineering.time_features      import build_time_features
from src.feature_engineering.email_features     import build_email_features
from src.feature_engineering.device_features    import build_device_features, get_device_encoders, apply_device_encoders
from src.feature_engineering.velocity_features  import build_velocity_features_offline
from src.feature_engineering.d_features         import build_d_features
from src.feature_engineering.c_features         import build_c_features
from src.feature_engineering.m_features         import build_m_features
from src.utils.reduce_mem                       import reduce_mem_usage


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_feature_engineering_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    artifacts_dir: str = "artifacts",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    train_df, test_df : raw merged DataFrames (from load_raw_data)
    cfg               : loaded config.yaml dict
    artifacts_dir     : where to save fitted encoders / imputers

    Returns
    -------
    datasets : dict with keys 'tree', 'catboost', 'vae', 'bilstm'
    encoders : fitted encoder objects (save these for inference)
    """
    art_path = Path(artifacts_dir) / "checkpoints"
    art_path.mkdir(parents=True, exist_ok=True)

    cat_cols = cfg["feature_engineering"]["catboost_cat_cols"]
    corr_thr = cfg["feature_engineering"]["v_col_corr_threshold"]
    n_pca    = cfg["feature_engineering"]["v_pca_components"]
    train_q  = cfg["split"]["train_quantile"]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 0: Sort by TransactionDT (MUST BE FIRST)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 0: Sort by TransactionDT ===")
    train_df = sort_by_time(train_df)
    test_df  = sort_by_time(test_df)

    train_df, val_df = time_split(train_df, train_q)
    logger.info("Train: {}  Val: {}  Test: {}", len(train_df), len(val_df), len(test_df))

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: UID construction
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 1: UID Features ===")
    for df_part in [train_df, val_df, test_df]:
        build_uid_features(df_part)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Encode M columns (before any NaN fills)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 2: M column encoding ===")
    for df_part in [train_df, val_df, test_df]:
        encode_m_columns(df_part, nan_value=-1)   # trees: -1
    # VAE version encoded separately later with nan_value=0

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Time features
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 3: Time Features ===")
    for df_part in [train_df, val_df, test_df]:
        build_time_features(df_part)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: D features (normalisation requires 'day')
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 4: D Features ===")
    # train_df, val_df, test_df are already separate splits — pass train_idx=None
    # (each is called on its own split, no cross-contamination)
    for df_part in [train_df, val_df, test_df]:
        build_d_features(df_part, train_idx=None)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Email + Device features
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 5: Email + Device Features ===")
    for df_part in [train_df, val_df, test_df]:
        build_email_features(df_part)
        build_device_features(df_part)

    device_encoders = get_device_encoders(train_df)  # fit on train only
    for df_part in [train_df, val_df, test_df]:       # apply consistently to all splits
        apply_device_encoders(df_part, device_encoders)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Frequency encoding (fit on train only)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 6: Frequency Encoding (train-only fit) ===")
    freq_cols = cfg["feature_engineering"]["freq_cols_tree"]
    freq_maps = fit_freq_encoder(train_df, cols=freq_cols)

    for df_part in [train_df, val_df, test_df]:
        apply_freq_encoder(df_part, freq_maps)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: UID aggregations (train stats only → apply to all)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 7: UID Aggregations ===")
    # Concatenate train+val+test, but only compute stats on train rows
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    train_idx_global = np.arange(len(train_df))
    all_df = build_uid_aggregations(all_df, train_idx=train_idx_global)
    all_df = reduce_mem_usage(all_df, verbose=False)

    # Split back
    n_train = len(train_df)
    n_val   = len(val_df)
    train_df = all_df.iloc[:n_train].copy()
    val_df   = all_df.iloc[n_train:n_train + n_val].copy()
    test_df  = all_df.iloc[n_train + n_val:].copy()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8: Amount features (requires uid_amt_mean)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 8: Amount Features ===")
    for df_part in [train_df, val_df, test_df]:
        build_amount_features(df_part)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9: C features
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 9: C Features ===")
    # Call on each split independently — no leakage possible
    for df_part in [train_df, val_df, test_df]:
        build_c_features(df_part, train_idx=None)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 10: M aggregation features (M column already encoded in STEP 2)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 10: M Aggregation Features ===")
    for df_part in [train_df, val_df, test_df]:
        build_m_features(df_part)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 11: Velocity features (offline batch — NOT for VAE)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 11: Velocity Features (offline) ===")
    # Pass history sequentially so val and test get correct gap distributions
    train_df, history = build_velocity_features_offline(train_df, return_history=True)
    val_df, history   = build_velocity_features_offline(val_df, uid_history=history, return_history=True)
    test_df           = build_velocity_features_offline(test_df, uid_history=history)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 12: V column reduction
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 12: V Column Reduction ===")
    v_groups = group_v_cols_by_nan(train_df)
    kept_v_cols = reduce_v_by_correlation(train_df, v_groups, corr_threshold=corr_thr)

    # Add uid-V aggregations (top 8 by SHAP — use defaults)
    all_df2 = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df2 = add_uid_v_aggregations(all_df2, train_idx=np.arange(len(train_df)))
    all_df2 = reduce_mem_usage(all_df2, verbose=False)
    train_df = all_df2.iloc[:n_train].copy()
    val_df   = all_df2.iloc[n_train:n_train + n_val].copy()
    test_df  = all_df2.iloc[n_train + n_val:].copy()

    # PCA-V for DL models
    v_pcas  = fit_v_pca(train_df, v_groups, n_components=n_pca)
    train_df, pca_cols = apply_v_pca(train_df, v_groups, v_pcas, "train")
    val_df,   _        = apply_v_pca(val_df,   v_groups, v_pcas, "val")
    test_df,  _        = apply_v_pca(test_df,  v_groups, v_pcas, "test")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 13: NaN fills for each model type
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=== STEP 13: Model-specific NaN fills ===")

    # 1. Extract VAE features first (while NaNs are still present and M cols are encoded with -1)
    feat_cfg_path = Path("configs/features.yaml")
    with open(feat_cfg_path) as f:
        feat_cfg = yaml.safe_load(f)
    vae_cols = (
        feat_cfg["vae_features"]["include"]
        + [c for c in pca_cols]
        + ["C1","C2","C3","C4","C5","C6","C7","C8","C9","C11","C13","C14"]
        + ["M1","M2","M3","M4","M5","M6","M7","M8","M9"]
    )
    vae_cols = [c for c in vae_cols if c in train_df.columns]

    vae_train = train_df[vae_cols].copy()
    vae_val   = val_df[vae_cols].copy()
    vae_test  = test_df[vae_cols].copy()
    for df_part in [vae_train, vae_val, vae_test]:
        for m in ["M1","M2","M3","M4","M5","M6","M7","M8","M9"]:
            if m in df_part.columns:
                df_part[m] = df_part[m].replace(-1, 0)

    median_imp = fit_median_imputer(vae_train.values)
    vae_train_arr = apply_median_imputer(median_imp, vae_train.values, "vae_train")
    vae_val_arr   = apply_median_imputer(median_imp, vae_val.values,   "vae_val")
    vae_test_arr  = apply_median_imputer(median_imp, vae_test.values,  "vae_test")

    # ── CRITICAL: Make deep copies BEFORE any in-place NaN fills ─────────────
    # tree and catboost need different NaN strategies; they must not share objects
    tree_train = train_df.copy()
    tree_val   = val_df.copy()
    tree_test  = test_df.copy()

    cb_train = train_df.copy()
    cb_val   = val_df.copy()
    cb_test  = test_df.copy()

    # Fill NaN per model type on their own independent copy
    fill_nan_for_trees(tree_train, exclude_cols=["isFraud"])
    fill_nan_for_trees(tree_val,   exclude_cols=["isFraud"])
    fill_nan_for_trees(tree_test,  exclude_cols=["isFraud"])

    fill_nan_for_catboost(cb_train, cat_cols)
    fill_nan_for_catboost(cb_val,   cat_cols)
    fill_nan_for_catboost(cb_test,  cat_cols)

    # ─────────────────────────────────────────────────────────────────────────
    # Save all artifacts
    # ─────────────────────────────────────────────────────────────────────────
    encoders = {
        "freq_maps":      freq_maps,
        "device_encoders": device_encoders,
        "v_groups":        v_groups,
        "kept_v_cols":     kept_v_cols,
        "v_pcas":          v_pcas,
        "pca_cols":        pca_cols,
        "vae_cols":        vae_cols,
        "median_imputer":  median_imp,
        "cat_cols":        cat_cols,
    }
    with open(art_path / "feature_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    logger.info("Encoders saved to {}", art_path / "feature_encoders.pkl")

    datasets = {
        "tree":  {"train": tree_train, "val": tree_val, "test": tree_test},
        "catboost": {"train": cb_train, "val": cb_val, "test": cb_test},
        "vae":  {
            "train": vae_train_arr, "val": vae_val_arr, "test": vae_test_arr,
            "cols": vae_cols,
        },
        "bilstm": {
            # BiLSTM sequences built by models/bilstm/dataset.py
            "train_raw": tree_train, "val_raw": tree_val, "test_raw": tree_test,
        },
    }

    return datasets, encoders
