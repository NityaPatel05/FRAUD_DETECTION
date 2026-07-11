import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger


def inject_gnn_embeddings(df: pd.DataFrame, card_embs: np.ndarray, card_map: dict) -> pd.DataFrame:
    """Map the 64-dim card embeddings into the dataframe based on card1."""
    if card_embs is None or not card_map:
        return df

    # Invert the map: card1_value -> embedding index
    card_to_idx = {str(k): v for k, v in card_map.items()}
    
    # We create a fallback embedding of zeros
    emb_dim = card_embs.shape[1]
    fallback = np.zeros(emb_dim, dtype=np.float32)
    
    # Map the df's card1 to embeddings
    def get_emb(card_val):
        idx = card_to_idx.get(str(card_val))
        if idx is not None and idx < len(card_embs):
            return card_embs[idx]
        return fallback

    # Apply mapping
    card1_series = df["card1"] if "card1" in df.columns else pd.Series(["unknown"] * len(df))
    emb_matrix = np.stack(card1_series.apply(get_emb).values)
    
    # Create column names
    emb_cols = [f"gnn_emb_{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index)
    
    return pd.concat([df, emb_df], axis=1)


def main(args):
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    art_dir    = Path(cfg["paths"]["checkpoints"])
    tree_dir   = Path(cfg["paths"]["processed_tree"])
    cb_dir     = Path(cfg["paths"]["processed_catboost"])
    vae_dir    = Path(cfg["paths"]["processed_vae"])
    bilstm_dir = Path(cfg["paths"]["processed_bilstm"])

    retrain_full = getattr(args, "retrain_full", False)
    if retrain_full:
        logger.info("=" * 60)
        logger.info("FULL RETRAIN MODE — training on train + val combined")
        logger.info("=" * 60)

    # ── Load datasets ─────────────────────────────────────────────────────────
    logger.info("Loading processed datasets…")
    tree_train = pd.read_parquet(tree_dir   / "train.parquet")
    tree_val   = pd.read_parquet(tree_dir   / "val.parquet")
    cb_train   = pd.read_parquet(cb_dir     / "train.parquet")
    cb_val     = pd.read_parquet(cb_dir     / "val.parquet")
    bl_train   = pd.read_parquet(bilstm_dir / "train.parquet")
    bl_val     = pd.read_parquet(bilstm_dir / "val.parquet")

    X_vae_train = np.load(vae_dir / "X_train.npy")
    X_vae_val   = np.load(vae_dir / "X_val.npy")
    y_train     = np.load(vae_dir / "y_train.npy")
    y_val       = np.load(vae_dir / "y_val.npy")

    all_val_probs  = {}
    all_oof_probs  = {}

    # ── If full retrain: merge train + val into one combined dataset ──────────────
    if retrain_full:
        logger.info("Concatenating train + val for full retrain…")
        tree_train = pd.concat([tree_train, tree_val], ignore_index=True)
        cb_train   = pd.concat([cb_train,   cb_val],   ignore_index=True)
        bl_train   = pd.concat([bl_train,   bl_val],   ignore_index=True)
        X_vae_train = np.concatenate([X_vae_train, X_vae_val], axis=0)
        y_train     = np.concatenate([y_train, y_val], axis=0)
        # In full retrain mode there is no separate val_df to pass
        tree_val = None
        cb_val   = None
        bl_val   = None
        logger.info(
            "Full retrain sizes — tree: {}  cb: {}  bilstm: {}  vae_X: {}",
            len(tree_train), len(cb_train), len(bl_train), len(X_vae_train),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: VAE
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_vae:
        logger.info("\n{}\nSTAGE 1: VAE\n{}", "=" * 60, "=" * 60)
        from src.models.vae.train import train_vae
        vae_results = train_vae(
            X_vae_train, y_train,
            X_vae_val,   y_val,
            save_dir=str(art_dir),
        )
        vae_train_scores = vae_results["train_scores"]
        vae_val_scores   = vae_results["val_scores"]
        logger.info("VAE done. Val AUC-PR: {:.5f}", vae_results["val_auc_pr"])
    else:
        vae_train_scores = np.load(art_dir / "vae_train_scores.npy")
        vae_val_scores   = np.load(art_dir / "vae_val_scores.npy")
        if retrain_full:
            vae_train_scores = np.concatenate([vae_train_scores, vae_val_scores])
            vae_val_scores   = None
        logger.info("VAE skipped — loaded saved scores.")

    tree_train["vae_anomaly_score"] = vae_train_scores
    if tree_val is not None:
        tree_val["vae_anomaly_score"]   = vae_val_scores
    cb_train["vae_anomaly_score"]   = vae_train_scores
    if cb_val is not None:
        cb_val["vae_anomaly_score"]     = vae_val_scores


    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: GNN (Moved earlier so tree models can use embeddings)
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_gnn:
        logger.info("\nSTAGE 2: GNN")
        from src.models.gnn.train import train_gnn
        # GNN uses the original data (bl_train/bl_val has the right features)
        gnn_results = train_gnn(
            bl_train, bl_val, save_dir=str(art_dir), use_cv=not retrain_full
        )

        
        card_embs = gnn_results.get("card_embeddings")
        card_map  = gnn_results.get("card_map")
        logger.info("GNN done. Val AUC-PR: {:.5f}", gnn_results["val_auc_pr"])
    else:

        
        # Load saved embeddings
        card_embs = np.load(art_dir / "gnn_card_embeddings.npy")
        with open(art_dir / "gnn_card_map.pkl", "rb") as f:
            card_map = pickle.load(f)
        logger.info("GNN skipped — loaded saved probs and embeddings.")

    # Inject GNN embeddings into tree dataframes
    logger.info("Injecting 64-dim GNN card embeddings into tree training data…")
    tree_train = inject_gnn_embeddings(tree_train, card_embs, card_map)
    if tree_val is not None:
        tree_val = inject_gnn_embeddings(tree_val, card_embs, card_map)
    cb_train   = inject_gnn_embeddings(cb_train, card_embs, card_map)
    if cb_val is not None:
        cb_val = inject_gnn_embeddings(cb_val, card_embs, card_map)

    # ───────────────────────────────────────────────────────────────────────
    # STAGE 3: XGBoost
    # ───────────────────────────────────────────────────────────────────────
    if not args.skip_xgb:
        logger.info("\nSTAGE 3: XGBoost")
        from src.models.xgboost_model.train import train_xgboost
        xgb_results = train_xgboost(
            tree_train, tree_val, save_dir=str(art_dir),
            use_cv=not retrain_full,
        )
        all_val_probs["xgb_prob"] = xgb_results["val_preds"]
        all_oof_probs["xgb_prob"] = xgb_results["oof_preds"]
        if not retrain_full:
            logger.info("XGBoost done. OOF AUC-PR: {:.5f}", xgb_results["oof_auc_pr"])
        else:
            logger.info("XGBoost full retrain done. Model: artifacts/checkpoints/xgb_final.pkl")
    else:
        all_oof_probs["xgb_prob"] = np.load(art_dir / "xgb_oof_preds.npy")
        all_val_probs["xgb_prob"] = np.load(art_dir / "xgb_val_preds.npy")
        logger.info("XGBoost skipped — loaded saved OOF and Val preds.")

    # ───────────────────────────────────────────────────────────────────────
    # STAGE 4: LightGBM
    # ───────────────────────────────────────────────────────────────────────
    if not args.skip_lgbm:
        logger.info("\nSTAGE 4: LightGBM")
        from src.models.lightgbm_model.train import train_lightgbm
        lgbm_results = train_lightgbm(
            tree_train, tree_val, save_dir=str(art_dir),
            use_cv=not retrain_full,
        )
        all_val_probs["lgbm_prob"] = lgbm_results["val_preds"]
        all_oof_probs["lgbm_prob"] = lgbm_results["oof_preds"]
        if not retrain_full:
            logger.info("LightGBM done. OOF AUC-PR: {:.5f}", lgbm_results["oof_auc_pr"])
        else:
            logger.info("LightGBM full retrain done. Model: artifacts/checkpoints/lgbm_final.pkl")
    else:
        all_oof_probs["lgbm_prob"] = np.load(art_dir / "lgbm_oof_preds.npy")
        all_val_probs["lgbm_prob"] = np.load(art_dir / "lgbm_val_preds.npy")
        logger.info("LightGBM skipped — loaded saved OOF and Val preds.")

    # ───────────────────────────────────────────────────────────────────────
    # STAGE 5: CatBoost
    # ───────────────────────────────────────────────────────────────────────
    if not args.skip_catboost:
        logger.info("\nSTAGE 5: CatBoost")
        from src.models.catboost_model.train import train_catboost
        cb_results = train_catboost(
            cb_train, cb_val, save_dir=str(art_dir),
            use_cv=not retrain_full,
        )
        all_val_probs["catboost_prob"] = cb_results["val_preds"]
        all_oof_probs["catboost_prob"] = cb_results["oof_preds"]
        if not retrain_full:
            logger.info("CatBoost done. OOF AUC-PR: {:.5f}", cb_results["oof_auc_pr"])
        else:
            logger.info("CatBoost full retrain done. Model: artifacts/checkpoints/catboost_final.cbm")
    else:
        all_oof_probs["catboost_prob"] = np.load(art_dir / "catboost_oof_preds.npy")
        all_val_probs["catboost_prob"] = np.load(art_dir / "catboost_val_preds.npy")
        logger.info("CatBoost skipped — loaded saved OOF and Val preds.")


    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 6: Stacking Meta-Learner
    # ─────────────────────────────────────────────────────────────────────────
    if retrain_full:
        # Load existing CV meta-learner; use final model full-data preds to get stacking score
        logger.info("\nSTAGE 6: Stacking — loading CV meta-learner, generating full-data stacking scores")
        import pickle as _pkl
        from src.models.stacking.meta_learner import predict_meta_learner
        with open(art_dir / "meta_learner.pkl", "rb") as _f:
            _meta = _pkl.load(_f)
        meta_model  = _meta["model"]
        meta_scaler = _meta["scaler"]
        full_probs = {k: v for k, v in all_val_probs.items() if v is not None}
        stacking_full      = predict_meta_learner(meta_model, meta_scaler, full_probs)
        stacking_oof_full  = stacking_full
        stacking_val_preds = stacking_full
        logger.info("Stacking score generated for {} rows using CV meta-learner.", len(stacking_full))
    else:
        logger.info("\nSTAGE 6: Stacking Meta-Learner")
        n_train = len(tree_train)
        n_val   = len(tree_val)

        aligned_oof = {}
        aligned_val = {}
        for k, arr in all_oof_probs.items():
            aligned_oof[k] = arr[:n_train] if len(arr) >= n_train else np.full(n_train, arr.mean())
        for k, arr in all_val_probs.items():
            if arr is not None:
                aligned_val[k] = arr[:n_val] if len(arr) >= n_val else np.full(n_val, arr.mean())

        from src.models.stacking.meta_learner import train_meta_learner

        valid_idx = aligned_oof["xgb_prob"] >= 0
        aligned_oof_valid = {k: v[valid_idx] for k, v in aligned_oof.items()}
        y_train_valid = y_train[:n_train][valid_idx]

        meta_results = train_meta_learner(
            train_probs=aligned_oof_valid,
            y_train=y_train_valid,
            val_probs=aligned_val,
            y_val=y_val[:n_val],
            save_dir=str(art_dir),
        )

        stacking_oof_full = np.zeros(n_train)
        stacking_oof_full[valid_idx] = meta_results.get("oof_preds", aligned_oof_valid["xgb_prob"])
        stacking_val_preds = meta_results["val_preds"]

        logger.info(
            "Stacking — Val AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}",
            meta_results["metrics"]["val_auc_roc"],
            meta_results["metrics"]["val_auc_pr"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 7: BiLSTM — Sequential Meta-Learner
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_bilstm:
        logger.info("\nSTAGE 7: BiLSTM Sequential Meta-Learner")

        n_bl_train = len(bl_train)

        if "TransactionID" in bl_train.columns and "TransactionID" in tree_train.columns:
            tid_to_stacking = dict(zip(tree_train["TransactionID"].values, stacking_oof_full))
            bl_train["calibrated_stacking_score"] = (
                bl_train["TransactionID"].map(tid_to_stacking).fillna(stacking_oof_full.mean())
            )
        else:
            logger.warning("TransactionID not available — using positional alignment.")
            bl_train["calibrated_stacking_score"] = stacking_oof_full[:n_bl_train]

        if not retrain_full and bl_val is not None:
            n_bl_val = len(bl_val)
            if "TransactionID" in bl_val.columns:
                tid_to_stacking_val = dict(zip(tree_train["TransactionID"].values, stacking_val_preds))
                bl_val["calibrated_stacking_score"] = (
                    bl_val["TransactionID"].map(tid_to_stacking_val).fillna(stacking_val_preds.mean())
                )
            else:
                bl_val["calibrated_stacking_score"] = stacking_val_preds[:n_bl_val]

        from src.models.bilstm.train import train_bilstm
        bl_results = train_bilstm(
            bl_train,
            bl_val if not retrain_full else bl_train,  # dummy val in full retrain
            save_dir=str(art_dir),
            use_cv=not retrain_full,
            cv_oof_path=str(art_dir / "bilstm_oof_preds.npy"),
            cv_y_train_path=str(vae_dir / "y_train.npy"),
        )
        all_val_probs["bilstm_prob"] = bl_results["val_probs"]
        all_oof_probs["bilstm_prob"] = bl_results["oof_preds"]
        if not retrain_full:
            logger.info("BiLSTM done. Val AUC-PR: {:.5f}", bl_results["val_auc_pr"])
        else:
            logger.info("BiLSTM full retrain done. Model: artifacts/checkpoints/bilstm_final.pt")
    else:
        logger.info("BiLSTM skipped.")

    logger.info("=== Training Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IEEE-CIS fraud detection pipeline")
    parser.add_argument("--skip-vae",      action="store_true", help="Skip VAE training (load saved)")
    parser.add_argument("--skip-xgb",      action="store_true", help="Skip XGBoost training")
    parser.add_argument("--skip-lgbm",     action="store_true", help="Skip LightGBM training")
    parser.add_argument("--skip-catboost", action="store_true", help="Skip CatBoost training")
    parser.add_argument("--skip-gnn",      action="store_true", help="Skip GNN training")
    parser.add_argument("--skip-bilstm",   action="store_true", help="Skip BiLSTM training")
    parser.add_argument(
        "--retrain-full",
        action="store_true",
        help=(
            "Full retrain mode: concatenates train+val, trains all base models on combined data "
            "with fixed iterations (no early stopping), uses existing CV meta-learner, "
            "and retrains BiLSTM with CV stacking score."
        ),
    )
    args = parser.parse_args()
    main(args)