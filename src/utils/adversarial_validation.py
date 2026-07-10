from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from loguru import logger


def run_adversarial_validation(
    train_df:       pd.DataFrame,
    test_df:        pd.DataFrame,
    feature_cols:   List[str],
    threshold:      float = 0.70,
    n_folds:        int   = 3,
    sample_size:    int   = 50_000,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Train a classifier to distinguish train vs test samples.
    Compute per-feature adversarial AUC.
    Return (safe_features, full_report_df).

    Parameters
    ----------
    train_df, test_df : DataFrames with feature_cols
    feature_cols      : list of feature names to evaluate
    threshold         : drop feature if adversarial AUC > threshold
    n_folds           : CV folds for adversarial classifier
    sample_size       : subsample size (full dataset can be slow)
    """
    logger.info(
        "Running adversarial validation on {} features (threshold={})…",
        len(feature_cols), threshold,
    )

    # Sub-sample for speed
    n_train = min(sample_size, len(train_df))
    n_test  = min(sample_size, len(test_df))
    tr_sub  = train_df[feature_cols].sample(n=n_train, random_state=42).fillna(-999)
    te_sub  = test_df[feature_cols].sample(n=n_test,  random_state=42).fillna(-999)

    # Build adversarial dataset: 1 = test, 0 = train
    X_adv = pd.concat([tr_sub, te_sub], ignore_index=True)
    y_adv = np.array([0] * n_train + [1] * n_test)

    # ── Full model adversarial AUC (sanity check) ─────────────────────────────
    from lightgbm import LGBMClassifier
    clf    = LGBMClassifier(n_estimators=200, num_leaves=64, random_state=42, verbose=-1)
    cv     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    for tr_idx, vl_idx in cv.split(X_adv, y_adv):
        clf.fit(X_adv.iloc[tr_idx], y_adv[tr_idx])
        preds = clf.predict_proba(X_adv.iloc[vl_idx])[:, 1]
        scores.append(roc_auc_score(y_adv[vl_idx], preds))
    full_adv_auc = np.mean(scores)
    logger.info("Full adversarial AUC: {:.4f}  (0.5 = no shift, 1.0 = extreme shift)", full_adv_auc)

    # ── Per-feature adversarial AUC ───────────────────────────────────────────
    results = []
    for col in feature_cols:
        x_col = X_adv[[col]].values
        col_scores = []
        for tr_idx, vl_idx in cv.split(x_col, y_adv):
            try:
                clf_single = LGBMClassifier(
                    n_estimators=50, num_leaves=8, random_state=42, verbose=-1
                )
                clf_single.fit(x_col[tr_idx], y_adv[tr_idx])
                preds = clf_single.predict_proba(x_col[vl_idx])[:, 1]
                col_scores.append(roc_auc_score(y_adv[vl_idx], preds))
            except Exception:
                col_scores.append(0.5)
        col_auc = np.mean(col_scores)
        results.append({"feature": col, "adversarial_auc": col_auc})

    report_df = pd.DataFrame(results).sort_values("adversarial_auc", ascending=False)

    # ── Flag and drop high-adversarial features ───────────────────────────────
    dropped = report_df[report_df["adversarial_auc"] > threshold]["feature"].tolist()
    safe    = [c for c in feature_cols if c not in dropped]

    logger.info(
        "Adversarial validation: {} features dropped (AUC > {}): {}",
        len(dropped), threshold, dropped[:10],
    )
    logger.info("Safe features remaining: {}", len(safe))

    return safe, report_df
