from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, average_precision_score
from loguru import logger

from src.preprocessing.sort_and_split import expanding_window_folds
from src.models.xgboost_model.train import EXCLUDE_COLS


# ── Hyperparameters ──────────────────────────────────────────────────────────
CATBOOST_PARAMS = {
    "iterations":          5000,
    "depth":               7,
    "learning_rate":       0.02474450334028733,
    "l2_leaf_reg":         7.657823007588947,
    "bagging_temperature": 1.5438737609577544,
    "random_strength":     1.0,
    "border_count":        128,
    "scale_pos_weight":    27,
    "nan_mode":            "Min",
    "eval_metric":         "PRAUC",
    "loss_function":       "Logloss",
    "task_type":           "GPU",
    "random_seed":         42,
    "verbose":             500,
    "early_stopping_rounds": 200,
    "use_best_model":      True,
}

CATBOOST_SEARCH_SPACE = {
    "depth":               ("int",   5, 10),
    "learning_rate":       ("float", 0.01, 0.1, True),
    "l2_leaf_reg":         ("float", 1.0, 10.0),
    "bagging_temperature": ("float", 0.0, 2.0),
}

# Final full-dataset retraining: avg bestIteration across 5 folds × 1.10
# (1803+1275+1917+2531+1860) / 5 = 1877.2  →  round(1877 × 1.10) = 2065
CATBOOST_FINAL_ITERATIONS = 2065

# Categorical columns CatBoost handles natively
CATBOOST_CAT_COLS = [
    "card4", "card6", "ProductCD", "DeviceType", "P_emaildomain",
]


# ── Training ──────────────────────────────────────────────────────────────────

def get_catboost_feature_cols(df: pd.DataFrame, cat_cols: List[str]) -> List[str]:
    """Return feature cols: numeric + raw string cat cols."""
    all_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and (df[c].dtype != object or c in cat_cols)
    ]
    return all_cols


def train_catboost(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    cat_cols: List[str] = CATBOOST_CAT_COLS,
    n_folds: int = 5,
    save_dir: str = "artifacts/checkpoints",
    use_cv: bool = True,
    final_iterations: Optional[int] = None,
) -> dict:
    """
    Train CatBoost with expanding-window CV.
    Uses CatBoost Pool with explicit cat_features for native encoding.

    Parameters
    ----------
    use_cv           : if False, single model on full data (full retrain mode)
    final_iterations : iterations when use_cv=False. Defaults to CATBOOST_FINAL_ITERATIONS.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    feature_cols  = get_catboost_feature_cols(train_df, cat_cols)
    present_cats  = [c for c in cat_cols if c in feature_cols]
    cat_feat_idx  = [feature_cols.index(c) for c in present_cats]

    X = train_df[feature_cols]
    y = train_df["isFraud"].values

    logger.info(
        "CatBoost training: {} features ({} cat), {} samples, {:.2%} fraud.",
        len(feature_cols), len(present_cats), len(X), y.mean(),
    )

    # ── Full-dataset retrain mode (no CV, fixed iterations) ──────────────────
    if not use_cv:
        n_iter = final_iterations or CATBOOST_FINAL_ITERATIONS
        logger.info(
            "CatBoost FULL RETRAIN mode — single model, iterations={} (no early stopping).", n_iter
        )
        # Build params without early_stopping_rounds and use_best_model
        params = {k: v for k, v in CATBOOST_PARAMS.items()
                  if k not in ("early_stopping_rounds", "use_best_model")}
        params["iterations"]    = n_iter
        params["use_best_model"] = False

        full_pool = Pool(X, y, cat_features=cat_feat_idx)
        model = CatBoostClassifier(**params)
        model.fit(full_pool)
        model.save_model(str(save_path / "catboost_final.cbm"))
        logger.info("CatBoost final model saved to {}", save_path / "catboost_final.cbm")

        # In full retrain mode, return predictions on the full dataset
        X_eval = val_df[feature_cols] if val_df is not None else X
        val_pool = Pool(X_eval, cat_features=cat_feat_idx)
        val_preds = model.predict_proba(val_pool)[:, 1]
        logger.info("CatBoost final model — predictions for {} rows.", len(val_preds))

        np.save(save_path / "catboost_final_preds.npy", val_preds if val_preds is not None else np.array([]))
        return {
            "models":       [model],
            "oof_preds":    None,
            "val_preds":    val_preds,
            "cv_auc_roc":   [],
            "cv_auc_pr":    [],
            "oof_auc_roc":  None,
            "oof_auc_pr":   None,
            "feature_cols": feature_cols,
            "cat_feat_idx": cat_feat_idx,
        }

    # ── Standard CV mode ─────────────────────────────────────────────────────
    models    = []
    cv_aucs   = []
    cv_auprs  = []
    oof_preds    = np.zeros(len(y))
    oof_val_mask = np.zeros(len(y), dtype=bool)  # boolean mask — avoids oof_preds>0 heuristic

    folds = expanding_window_folds(train_df, n_folds=n_folds)

    for fold_idx, (tr_idx, vl_idx) in enumerate(folds):
        logger.info("Fold {}/{}: train={}, val={}", fold_idx+1, len(folds), len(tr_idx), len(vl_idx))

        tr_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_feat_idx)
        vl_pool = Pool(X.iloc[vl_idx], y[vl_idx], cat_features=cat_feat_idx)

        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(tr_pool, eval_set=vl_pool, use_best_model=True)

        preds_vl = model.predict_proba(vl_pool)[:, 1]
        oof_preds[vl_idx]    = preds_vl
        oof_val_mask[vl_idx] = True   # track which rows have valid OOF predictions

        fold_auc  = roc_auc_score(y[vl_idx], preds_vl)
        fold_aupr = average_precision_score(y[vl_idx], preds_vl)
        cv_aucs.append(fold_auc)
        cv_auprs.append(fold_aupr)
        logger.info("  AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}", fold_auc, fold_aupr)

        models.append(model)
        model.save_model(str(save_path / f"catboost_fold{fold_idx}.cbm"))

    val_indices = oof_val_mask   # fix: was oof_preds > 0 which misses near-zero fraud probs
    oof_auc  = roc_auc_score(y[val_indices], oof_preds[val_indices])
    oof_aupr = average_precision_score(y[val_indices], oof_preds[val_indices])
    logger.info("CatBoost OOF — AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}", oof_auc, oof_aupr)

    val_preds = None
    if val_df is not None:
        X_val = val_df[feature_cols]
        val_pool = Pool(X_val, cat_features=cat_feat_idx)
        val_preds = np.mean([m.predict_proba(val_pool)[:, 1] for m in models], axis=0)
        logger.info(
            "CatBoost Val — AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}",
            roc_auc_score(val_df["isFraud"].values, val_preds),
            average_precision_score(val_df["isFraud"].values, val_preds),
        )

    np.save(save_path / "catboost_oof_preds.npy", oof_preds)
    if val_preds is not None:
        np.save(save_path / "catboost_val_preds.npy", val_preds)

    return {
        "models":       models,
        "oof_preds":    oof_preds,
        "val_preds":    val_preds,
        "cv_auc_roc":   cv_aucs,
        "cv_auc_pr":    cv_auprs,
        "oof_auc_roc":  oof_auc,
        "oof_auc_pr":   oof_aupr,
        "feature_cols": feature_cols,
        "cat_feat_idx": cat_feat_idx,
    }
