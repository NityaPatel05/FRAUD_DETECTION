from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from loguru import logger

from src.models.xgboost_model.hyperparams import XGB_PARAMS, XGB_FIT_PARAMS, XGB_FINAL_N_ESTIMATORS
from src.preprocessing.sort_and_split import expanding_window_folds


# ── Feature exclusions ────────────────────────────────────────────────────────
EXCLUDE_COLS = [
    "TransactionID", "TransactionDT", "isFraud",
    "uid1", "uid2", "uid3", "uid_magic",       # string UIDs (use freq-encoded versions)
    "card4", "card6", "ProductCD", "DeviceType", "P_emaildomain",  # raw strings
    "device_os", "device_browser",              # raw strings
    "P_email_tld", "R_email_tld",               # raw strings
    "day", "hour", "weekday", "week", "month",  # raw (cyclic versions kept)
]


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return numeric feature columns suitable for XGBoost."""
    return [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and df[c].dtype != object
    ]


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    n_folds: int = 5,
    save_dir: str = "artifacts/checkpoints",
    use_cv: bool = True,
    final_n_estimators: Optional[int] = None,
) -> dict:
    """
    Train XGBoost with expanding-window CV.

    Parameters
    ----------
    train_df  : training dataframe (NaN filled with -999)
    val_df    : optional held-out validation set for final eval
    n_folds   : number of CV folds
    save_dir  : directory to save models and OOF predictions
    use_cv    : if False, train single model on full train_df (full retrain mode)
    final_n_estimators : n_estimators to use when use_cv=False.
                         Defaults to XGB_FINAL_N_ESTIMATORS if not provided.

    Returns
    -------
    results: dict with keys 'oof_preds', 'val_preds', 'feature_importance', 'cv_scores'
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_cols(train_df)
    target_col   = "isFraud"

    X = train_df[feature_cols].values
    y = train_df[target_col].values

    logger.info(
        "XGBoost training: {} features, {} samples, {:.2%} fraud rate.",
        len(feature_cols), len(X), y.mean(),
    )

    # ── Full-dataset retrain mode (no CV, fixed iterations) ──────────────────
    if not use_cv:
        n_est = final_n_estimators or XGB_FINAL_N_ESTIMATORS
        logger.info(
            "XGBoost FULL RETRAIN mode — single model, n_estimators={} (no early stopping).", n_est
        )
        # Build params without early_stopping_rounds
        params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
        params["n_estimators"] = n_est

        model = xgb.XGBClassifier(**params)
        model.fit(X, y, **XGB_FIT_PARAMS)

        with open(save_path / "xgb_final.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info("XGBoost final model saved to {}", save_path / "xgb_final.pkl")

        # Feature importance
        importances = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        importances.to_csv(save_path / "xgb_final_feature_importance.csv")

        # Predict on val_df if provided, else on the full training set
        # (in full retrain mode, the meta-learner needs predictions on the full dataset)
        X_eval = val_df[feature_cols].values if val_df is not None else X
        val_preds = model.predict_proba(X_eval)[:, 1]
        logger.info(
            "XGBoost final model — predictions computed for {} rows.", len(val_preds)
        )

        return {
            "models":             [model],
            "oof_preds":          None,
            "val_preds":          val_preds,
            "feature_importance": importances,
            "cv_auc_roc":         [],
            "cv_auc_pr":          [],
            "oof_auc_roc":        None,
            "oof_auc_pr":         None,
            "feature_cols":       feature_cols,
        }

    # ── Standard CV mode ─────────────────────────────────────────────────────
    models      = []
    cv_aucs     = []
    cv_auprs    = []
    importances = pd.Series(0.0, index=feature_cols)
    oof_preds   = np.zeros(len(X))
    oof_val_mask = np.zeros(len(X), dtype=bool)

    folds = expanding_window_folds(train_df, n_folds=n_folds)
    for fold_idx, (tr_idx, vl_idx) in enumerate(folds):
        logger.info(
            "Fold {}/{}: train={}, val={}",
            fold_idx + 1, len(folds), len(tr_idx), len(vl_idx),
        )

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_vl, y_vl = X[vl_idx], y[vl_idx]

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            **XGB_FIT_PARAMS,
        )

        preds_vl = model.predict_proba(X_vl)[:, 1]
        oof_preds[vl_idx] = preds_vl
        oof_val_mask[vl_idx] = True

        fold_auc  = roc_auc_score(y_vl, preds_vl)
        fold_aupr = average_precision_score(y_vl, preds_vl)
        cv_aucs.append(fold_auc)
        cv_auprs.append(fold_aupr)
        logger.info("  AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}", fold_auc, fold_aupr)

        importances += pd.Series(
            model.feature_importances_, index=feature_cols
        )

        models.append(model)
        with open(save_path / f"xgb_fold{fold_idx}.pkl", "wb") as f:
            pickle.dump(model, f)

    # OOF scores
    val_indices = oof_val_mask
    oof_auc  = roc_auc_score(y[val_indices], oof_preds[val_indices])
    oof_aupr = average_precision_score(y[val_indices], oof_preds[val_indices])
    logger.info(
        "XGBoost OOF — AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}",
        oof_auc, oof_aupr,
    )

    # Val set predictions (average across all fold models)
    val_preds = None
    if val_df is not None:
        X_val = val_df[feature_cols].values
        val_preds = np.mean(
            [m.predict_proba(X_val)[:, 1] for m in models], axis=0
        )
        val_auc  = roc_auc_score(val_df["isFraud"].values, val_preds)
        val_aupr = average_precision_score(val_df["isFraud"].values, val_preds)
        logger.info(
            "XGBoost Val — AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}",
            val_auc, val_aupr,
        )

    # Feature importance
    importances /= len(folds)
    importances = importances.sort_values(ascending=False)
    importances.to_csv(save_path / "xgb_feature_importance.csv")

    # Save OOF and metadata
    np.save(save_path / "xgb_oof_preds.npy", oof_preds)
    if val_preds is not None:
        np.save(save_path / "xgb_val_preds.npy", val_preds)
    pd.DataFrame({"col": feature_cols}).to_csv(save_path / "xgb_feature_cols.csv", index=False)

    return {
        "models":             models,
        "oof_preds":          oof_preds,
        "val_preds":          val_preds,
        "feature_importance": importances,
        "cv_auc_roc":         cv_aucs,
        "cv_auc_pr":          cv_auprs,
        "oof_auc_roc":        oof_auc,
        "oof_auc_pr":         oof_aupr,
        "feature_cols":       feature_cols,
    }


def predict_xgboost(
    models: list,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """Ensemble predict — average probabilities across all fold models."""
    X = df[feature_cols].values
    probs = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return probs
