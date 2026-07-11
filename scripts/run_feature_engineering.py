import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from loguru import logger

from src.feature_engineering.pipeline import run_feature_engineering_pipeline
from src.utils.forward_feature_selection import run_d_column_selection


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    raw_processed = Path("data/processed")
    art_dir       = Path(cfg["paths"]["checkpoints"])

    logger.info("=== Feature Engineering Pipeline ===")

    # ── Load preprocessed splits ──────────────────────────────────────────────
    logger.info("Loading preprocessed splits…")
    train_df = pd.read_parquet(raw_processed / "train_raw.parquet")
    val_df   = pd.read_parquet(raw_processed / "val_raw.parquet")
    test_df  = pd.read_parquet(raw_processed / "test_raw.parquet")

    # ── Run main pipeline ─────────────────────────────────────────────────────
    datasets, encoders = run_feature_engineering_pipeline(
        train_df=train_df,
        test_df=pd.concat([val_df, test_df], ignore_index=True),
        cfg=cfg,
        artifacts_dir=str(art_dir.parent),
    )

    # ── Forward D column selection (validate which D_norm cols to keep) ───────
    logger.info("Running forward D column selection…")
    tree_train = datasets["tree"]["train"]
    tree_val   = datasets["tree"]["val"]
    kept_d     = run_d_column_selection(tree_train, tree_val)
    logger.info("Validated D_norm columns to use: {}", kept_d)

    import pickle
    with open(art_dir / "kept_d_cols.pkl", "wb") as f:
        pickle.dump(kept_d, f)

    # ── Save processed datasets ───────────────────────────────────────────────
    logger.info("Saving processed datasets…")

    # Tree models
    tree_dir = Path(cfg["paths"]["processed_tree"])
    tree_dir.mkdir(parents=True, exist_ok=True)
    datasets["tree"]["train"].to_parquet(tree_dir / "train.parquet", index=False)
    datasets["tree"]["val"].to_parquet(tree_dir   / "val.parquet",   index=False)
    datasets["tree"]["test"].to_parquet(tree_dir  / "test.parquet",  index=False)
    logger.info("Tree model data saved: {}", tree_dir)

    # CatBoost
    cb_dir = Path(cfg["paths"]["processed_catboost"])
    cb_dir.mkdir(parents=True, exist_ok=True)
    datasets["catboost"]["train"].to_parquet(cb_dir / "train.parquet", index=False)
    datasets["catboost"]["val"].to_parquet(cb_dir   / "val.parquet",   index=False)
    datasets["catboost"]["test"].to_parquet(cb_dir  / "test.parquet",  index=False)
    logger.info("CatBoost data saved: {}", cb_dir)

    # VAE
    vae_dir = Path(cfg["paths"]["processed_vae"])
    vae_dir.mkdir(parents=True, exist_ok=True)
    np.save(vae_dir / "X_train.npy", datasets["vae"]["train"])
    np.save(vae_dir / "X_val.npy",   datasets["vae"]["val"])
    np.save(vae_dir / "X_test.npy",  datasets["vae"]["test"])
    np.save(vae_dir / "y_train.npy", tree_train["isFraud"].values)
    np.save(vae_dir / "y_val.npy",   tree_val["isFraud"].values)
    logger.info("VAE data saved: {}", vae_dir)

    # BiLSTM raw (sequences built during training by dataset.py)
    bilstm_dir = Path(cfg["paths"]["processed_bilstm"])
    bilstm_dir.mkdir(parents=True, exist_ok=True)
    datasets["bilstm"]["train_raw"].to_parquet(bilstm_dir / "train.parquet", index=False)
    datasets["bilstm"]["val_raw"].to_parquet(bilstm_dir   / "val.parquet",   index=False)
    datasets["bilstm"]["test_raw"].to_parquet(bilstm_dir  / "test.parquet",  index=False)
    logger.info("BiLSTM data saved: {}", bilstm_dir)

    logger.info("=== Feature Engineering Complete ===")
    logger.info("Total features (tree): {}", len([c for c in tree_train.columns if tree_train[c].dtype != object]))


if __name__ == "__main__":
    main()
