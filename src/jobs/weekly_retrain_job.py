from __future__ import annotations

import os
import pickle
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from src.db.schema import (
    ModelVersion,
    TrainingTransaction,
    SessionLocal,
    create_all_tables,
    get_engine,
)
from src.models.stacking.meta_learner import train_meta_learner


# ─────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ─────────────────────────────────────────────────────────────────────────────

MIN_NEW_LABELS    = 500     # skip retrain if fewer new labels available
PROMOTE_THRESHOLD = 0.003   # new model must beat champion AUC-PR by this margin
SHADOW_WINDOW_DAYS = 2      # run shadow for 2 days before deciding
NO_IMPROVE_MAX    = 4       # retire champion after 4 failed promote attempts


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load new confirmed labels from DB
# ─────────────────────────────────────────────────────────────────────────────

def load_new_labels(engine, since: datetime) -> pd.DataFrame:
    """
    Pull confirmed fraud labels from the DB that arrived after `since`.

    Returns a DataFrame with columns:
        transaction_id, uid, transaction_dt, transaction_amt,
        final_fraud_score, confirmed_fraud, labelled_at, ...
    """
    session = SessionLocal(engine)
    try:
        rows = (
            session.query(TrainingTransaction)
            .filter(
                TrainingTransaction.confirmed_fraud.isnot(None),
                TrainingTransaction.labelled_at > since,
            )
            .all()
        )
        if not rows:
            return pd.DataFrame()

        data = [
            {
                "transaction_id":   r.transaction_id,
                "uid":              r.uid,
                "TransactionDT":    r.transaction_dt,
                "TransactionAmt":   r.transaction_amt,
                "card1":            r.card1,
                "DeviceInfo":       r.device_info,
                "P_emaildomain":    r.p_emaildomain,
                "isFraud":          r.confirmed_fraud,
                "final_fraud_score": r.final_fraud_score,
                "labelled_at":      r.labelled_at,
            }
            for r in rows
        ]
        df = pd.DataFrame(data)
        logger.info("Loaded {} confirmed labels from DB (since={}).", len(df), since.date())
        return df
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Retrain XGBoost folds + stacking
# ─────────────────────────────────────────────────────────────────────────────

def retrain_xgb_and_stacking(
    base_df:    pd.DataFrame,
    new_df:     pd.DataFrame,
    checkpoint_dir: str,
    save_dir:   str,
) -> dict:
    """
    Retrain XGBoost fold models + stacking meta-learner on
    base_df ∪ new_df (new labels appended to the original training set).

    Returns dict with val_auc_roc, val_auc_pr, model handles.
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    # Combine original train + new confirmed labels
    # new_df already has isFraud = confirmed_fraud (set by caller)
    combined = pd.concat([base_df, new_df], ignore_index=True)
    combined = combined.sort_values("TransactionDT").reset_index(drop=True)

    # Time-aware split: last 20% as validation
    split_idx = int(len(combined) * 0.80)
    train_df  = combined.iloc[:split_idx]
    val_df    = combined.iloc[split_idx:]

    # Load existing feature columns list
    feat_df      = pd.read_csv(Path(checkpoint_dir) / "xgb_feature_cols.csv")
    feature_cols = feat_df["col"].tolist()
    present_train = [c for c in feature_cols if c in train_df.columns]
    present_val   = [c for c in feature_cols if c in val_df.columns]

    X_train = train_df[present_train].fillna(-999)
    y_train = train_df["isFraud"].values
    X_val   = val_df[present_val].fillna(-999)
    y_val   = val_df["isFraud"].values

    # Load existing XGBoost params from one of the saved fold models
    existing_path = sorted(Path(checkpoint_dir).glob("xgb_fold*.pkl"))[0]
    with open(existing_path, "rb") as f:
        existing_model = pickle.load(f)
    params = existing_model.get_params()

    # Train single model (not k-fold) for speed — production k-fold done on longer cadence
    new_xgb = xgb.XGBClassifier(**params, random_state=42)
    new_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_probs = new_xgb.predict_proba(X_val)[:, 1]
    val_auc   = roc_auc_score(y_val, val_probs)
    val_aupr  = average_precision_score(y_val, val_probs)

    logger.info(
        "Retrained XGBoost — Val AUC-ROC: {:.5f}  AUC-PR: {:.5f}",
        val_auc, val_aupr,
    )

    # Save to temp dir (only copied to checkpoints on promotion)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    new_model_path = Path(save_dir) / "xgb_retrained.pkl"
    with open(new_model_path, "wb") as f:
        pickle.dump(new_xgb, f)

    # Retrain stacking meta-learner with new XGB OOF probs
    # (use val probs as a proxy for OOF — acceptable for weekly cadence)
    val_probs_dict  = {"xgb_prob": val_probs}

    # Load LGBM and CatBoost val probs from existing checkpoints (static for now)
    lgbm_val_path = Path(checkpoint_dir) / "lgbm_val_preds.npy"
    cb_val_path   = Path(checkpoint_dir) / "catboost_val_preds.npy"
    if lgbm_val_path.exists():
        val_probs_dict["lgbm_prob"] = np.load(lgbm_val_path)[:len(y_val)]
    if cb_val_path.exists():
        val_probs_dict["catboost_prob"] = np.load(cb_val_path)[:len(y_val)]

    stacking_result = train_meta_learner(
        train_probs={k: v for k, v in val_probs_dict.items()},
        y_train=y_val,
        val_probs=val_probs_dict,
        y_val=y_val,
        save_dir=save_dir,
    )

    return {
        "xgb_model":    new_xgb,
        "val_auc_roc":  val_auc,
        "val_auc_pr":   val_aupr,
        "stacking_auc": stacking_result["metrics"]["val_auc_roc"],
        "save_dir":     save_dir,
        "n_train":      len(train_df),
        "n_val":        len(val_df),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Shadow comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_champion_vs_shadow(
    champion_dir: str,
    shadow_dir:   str,
    val_df:       pd.DataFrame,
) -> dict:
    """
    Run both champion and shadow models on val_df, compare AUC-PR.
    Returns dict with champion_aupr, shadow_aupr, promote=True/False.
    """
    feat_df      = pd.read_csv(Path(champion_dir) / "xgb_feature_cols.csv")
    feature_cols = feat_df["col"].tolist()
    present      = [c for c in feature_cols if c in val_df.columns]

    X_val = val_df[present].fillna(-999)
    y_val = val_df["isFraud"].values

    # Champion
    champ_paths = sorted(Path(champion_dir).glob("xgb_fold*.pkl"))
    champ_probs = np.zeros(len(X_val))
    for p in champ_paths:
        with open(p, "rb") as f:
            m = pickle.load(f)
        champ_probs += m.predict_proba(X_val)[:, 1]
    champ_probs /= max(len(champ_paths), 1)
    champion_aupr = average_precision_score(y_val, champ_probs)

    # Shadow
    shadow_path = Path(shadow_dir) / "xgb_retrained.pkl"
    with open(shadow_path, "rb") as f:
        shadow_model = pickle.load(f)
    shadow_probs = shadow_model.predict_proba(X_val)[:, 1]
    shadow_aupr  = average_precision_score(y_val, shadow_probs)

    promote = shadow_aupr > champion_aupr + PROMOTE_THRESHOLD
    logger.info(
        "Shadow comparison — champion_aupr={:.5f}  shadow_aupr={:.5f}  promote={}",
        champion_aupr, shadow_aupr, promote,
    )

    return {
        "champion_aupr": champion_aupr,
        "shadow_aupr":   shadow_aupr,
        "promote":       promote,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Promote shadow to champion
# ─────────────────────────────────────────────────────────────────────────────

def promote_shadow(
    shadow_dir:     str,
    champion_dir:   str,
    version_tag:    str,
    engine,
    session,
    shadow_aupr:    float,
    n_new_labels:   int,
    label_cutoff:   datetime,
    mlflow_run_id:  Optional[str] = None,
) -> None:
    """
    Copy shadow artifacts over the champion directory.
    Update model_versions table: retire old champion, mark new one as champion.
    """
    # Retire current champion in DB
    old_champ = (
        session.query(ModelVersion)
        .filter_by(is_champion=True)
        .one_or_none()
    )
    if old_champ:
        old_champ.is_champion = False
        old_champ.is_retired  = True
        old_champ.retired_at  = datetime.utcnow()
        logger.info("Retired champion: {}", old_champ.version_tag)

    # Copy shadow artifacts → champion directory
    for src_file in Path(shadow_dir).iterdir():
        dest = Path(champion_dir) / src_file.name
        shutil.copy2(src_file, dest)
    logger.info("Shadow artifacts copied to champion dir: {}", champion_dir)

    # Register new champion in DB
    new_champ = ModelVersion(
        version_tag     = version_tag,
        model_type      = "xgb_stacking",
        checkpoint_path = champion_dir,
        n_new_labels    = n_new_labels,
        label_cutoff_dt = label_cutoff,
        shadow_auc_roc  = shadow_aupr,
        val_auc_pr      = shadow_aupr,
        is_shadow       = False,
        is_champion     = True,
        is_retired      = False,
        promoted_at     = datetime.utcnow(),
        mlflow_run_id   = mlflow_run_id,
        trained_by      = "weekly_retrain_job",
    )
    session.add(new_champ)
    session.commit()
    logger.info("New champion registered: {} (AUC-PR={:.5f})", version_tag, shadow_aupr)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def weekly_retrain_job(
    config_path:   str = "configs/config.yaml",
    dry_run:       bool = False,
    lookback_days: int = 7,
) -> dict:
    """
    Weekly retrain with shadow deploy.

    Parameters
    ----------
    config_path   : Path to configs/config.yaml
    dry_run       : If True, does NOT write to DB or overwrite checkpoints.
    lookback_days : How many days of new labels to include.

    Returns
    -------
    dict with status, n_new_labels, promoted (bool), metrics
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    checkpoint_dir = cfg["paths"]["checkpoints"]
    db_url         = cfg["database"]["url"]
    since          = datetime.utcnow() - timedelta(days=lookback_days)

    # ── 1. DB setup ───────────────────────────────────────────────────────────
    engine  = get_engine(db_url)
    create_all_tables(engine)

    # ── 2. Load new labels ────────────────────────────────────────────────────
    new_labels_df = load_new_labels(engine, since=since)

    if len(new_labels_df) < MIN_NEW_LABELS:
        logger.info(
            "weekly_retrain_job: only {} new labels (need {}). Skipping.",
            len(new_labels_df), MIN_NEW_LABELS,
        )
        return {"status": "skipped", "n_new_labels": len(new_labels_df)}

    logger.info("weekly_retrain_job: {} new labels. Starting retrain…", len(new_labels_df))

    # ── 3. Load base training data ────────────────────────────────────────────
    base_path = Path(cfg["paths"]["processed_tree"]) / "train.parquet"
    if not base_path.exists():
        base_path = Path(cfg["paths"]["processed_tree"]) / "train.csv"
    if not base_path.exists():
        logger.error("Base training data not found at {}. Aborting.", base_path)
        return {"status": "error", "reason": "base data missing"}

    base_df = (
        pd.read_parquet(base_path)
        if str(base_path).endswith(".parquet")
        else pd.read_csv(base_path)
    )
    logger.info("Base training data: {} rows.", len(base_df))

    # ── 4. Retrain in a temp directory ────────────────────────────────────────
    shadow_dir = tempfile.mkdtemp(prefix="fraud_shadow_")
    try:
        result = retrain_xgb_and_stacking(
            base_df=base_df,
            new_df=new_labels_df,
            checkpoint_dir=checkpoint_dir,
            save_dir=shadow_dir,
        )
    except Exception as exc:
        logger.error("weekly_retrain_job: retrain failed — {}. Aborting.", exc)
        shutil.rmtree(shadow_dir, ignore_errors=True)
        return {"status": "error", "reason": str(exc)}

    # ── 5. Shadow comparison ──────────────────────────────────────────────────
    # Use the val split from retrain_xgb_and_stacking (last 20% of combined data)
    combined  = pd.concat([base_df, new_labels_df]).sort_values("TransactionDT")
    split_idx = int(len(combined) * 0.80)
    val_df    = combined.iloc[split_idx:].reset_index(drop=True)

    comparison = compare_champion_vs_shadow(
        champion_dir=checkpoint_dir,
        shadow_dir=shadow_dir,
        val_df=val_df,
    )

    promote = comparison["promote"]
    version_tag = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M')}"

    if dry_run:
        logger.info(
            "weekly_retrain_job [DRY RUN]: promote={} shadow_aupr={:.5f}",
            promote, comparison["shadow_aupr"],
        )
        shutil.rmtree(shadow_dir, ignore_errors=True)
        return {
            "status":       "dry_run",
            "promote":      promote,
            "n_new_labels": len(new_labels_df),
            "metrics":      comparison,
        }

    # ── 6. Promote or discard ─────────────────────────────────────────────────
    session = SessionLocal(engine)
    try:
        if promote:
            promote_shadow(
                shadow_dir=shadow_dir,
                champion_dir=checkpoint_dir,
                version_tag=version_tag,
                engine=engine,
                session=session,
                shadow_aupr=comparison["shadow_aupr"],
                n_new_labels=len(new_labels_df),
                label_cutoff=new_labels_df["labelled_at"].max(),
            )
        else:
            # Log the failed attempt
            failed_version = ModelVersion(
                version_tag     = version_tag + "_discarded",
                model_type      = "xgb_stacking",
                checkpoint_path = shadow_dir,
                n_new_labels    = len(new_labels_df),
                val_auc_pr      = comparison["shadow_aupr"],
                is_shadow       = False,
                is_champion     = False,
                is_retired      = True,
                trained_by      = "weekly_retrain_job",
            )
            session.add(failed_version)
            session.commit()
            logger.info(
                "Shadow did not beat champion (Δ={:.5f} < {:.5f}). Discarded.",
                comparison["shadow_aupr"] - comparison["champion_aupr"],
                PROMOTE_THRESHOLD,
            )

    except Exception as exc:
        session.rollback()
        logger.error("weekly_retrain_job: DB write failed — {}", exc)
        raise
    finally:
        session.close()
        shutil.rmtree(shadow_dir, ignore_errors=True)

    return {
        "status":       "promoted" if promote else "no_improve",
        "version_tag":  version_tag,
        "promote":      promote,
        "n_new_labels": len(new_labels_df),
        "metrics":      comparison,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weekly model retrain with shadow deploy")
    parser.add_argument("--config",        default="configs/config.yaml")
    parser.add_argument("--dry-run",       action="store_true", help="No DB/file writes")
    parser.add_argument("--lookback-days", type=int, default=7)
    args = parser.parse_args()

    result = weekly_retrain_job(
        config_path=args.config,
        dry_run=args.dry_run,
        lookback_days=args.lookback_days,
    )

    print(f"\nResult: {result['status']}")
    if "metrics" in result:
        m = result["metrics"]
        print(f"  Champion AUC-PR : {m.get('champion_aupr', 'N/A'):.5f}")
        print(f"  Shadow   AUC-PR : {m.get('shadow_aupr',   'N/A'):.5f}")
        print(f"  Promoted        : {result.get('promote', False)}")
    print(f"  New labels used : {result.get('n_new_labels', 0)}")
