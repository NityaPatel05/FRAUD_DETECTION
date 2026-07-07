from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from loguru import logger

from src.models.lightgbm_model.hyperparams import LGBM_PARAMS, LGBM_FINAL_N_ESTIMATORS
from src.preprocessing.sort_and_split import expanding_window_folds
from src.models.xgboost_model.train import get_feature_cols  # reuse same logic


def train_lightgbm(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    n_folds: int = 5,
    save_dir: str = "artifacts/checkpoints",
    use_cv: bool = True,
    final_n_estimators: Optional[int] = None,
) -> dict:
    """
    Train LightGBM with expanding-window CV.
    Same interface as train_xgboost() for easy swapping.

    Parameters
    ----------
    use_cv              : if False, train single model on full train_df (full retrain mode)
    final_n_estimators  : n_estimators when use_cv=False. Defaults to LGBM_FINAL_N_ESTIMATORS.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_cols(train_df)
    X = train_df[feature_cols].values
    y = train_df["isFraud"].values

    logger.info(
        "LightGBM training: {} features, {} samples, {:.2%} fraud rate.",
        len(feature_cols), len(X), y.mean(),
    )

    # ── Full-dataset retrain mode (no CV, fixed iterations, no early stopping) ───
    if not use_cv:
        n_est = final_n_estimators or LGBM_FINAL_N_ESTIMATORS
        logger.info(
            "LightGBM FULL RETRAIN mode — single model, n_estimators={} (no early stopping).", n_est
        )
        params = {**LGBM_PARAMS, "n_estimators": n_est}
        model  = lgb.LGBMClassifier(**params)
        # No eval_set — no early stopping
        model.fit(X, y)

        with open(save_path / "lgbm_final.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info("LightGBM final model saved to {}", save_path / "lgbm_final.pkl")

        importances = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        importances.to_csv(save_path / "lgbm_final_feature_importance.csv")

        # In full retrain mode, return predictions on the full dataset
        X_eval = val_df[feature_cols].values if val_df is not None else X
        val_preds = model.predict_proba(X_eval)[:, 1]
        logger.info(
            "LightGBM final model — predictions for {} rows.", len(val_preds)
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

        callbacks = [
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=500),
        ]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            callbacks=callbacks,
        )

        preds_vl       = model.predict_proba(X_vl)[:, 1]
        oof_preds[vl_idx] = preds_vl
        oof_val_mask[vl_idx] = True

        fold_auc  = roc_auc_score(y_vl, preds_vl)
        fold_aupr = average_precision_score(y_vl, preds_vl)
        cv_aucs.append(fold_auc)
        cv_auprs.append(fold_aupr)
        logger.info("  AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}", fold_auc, fold_aupr)

        importances += pd.Series(model.feature_importances_, index=feature_cols)
        models.append(model)

        with open(save_path / f"lgbm_fold{fold_idx}.pkl", "wb") as f:
            pickle.dump(model, f)

    val_indices = oof_val_mask
    oof_auc  = roc_auc_score(y[val_indices], oof_preds[val_indices])
    oof_aupr = average_precision_score(y[val_indices], oof_preds[val_indices])
    logger.info(
        "LightGBM OOF — AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}", oof_auc, oof_aupr,
    )

    val_preds = None
    if val_df is not None:
        X_val = val_df[feature_cols].values
        val_preds = np.mean(
            [m.predict_proba(X_val)[:, 1] for m in models], axis=0
        )
        val_auc  = roc_auc_score(val_df["isFraud"].values, val_preds)
        val_aupr = average_precision_score(val_df["isFraud"].values, val_preds)
        logger.info(
            "LightGBM Val — AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}", val_auc, val_aupr,
        )

    importances /= len(folds)
    importances = importances.sort_values(ascending=False)
    importances.to_csv(save_path / "lgbm_feature_importance.csv")
    np.save(save_path / "lgbm_oof_preds.npy", oof_preds)
    if val_preds is not None:
        np.save(save_path / "lgbm_val_preds.npy", val_preds)

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
