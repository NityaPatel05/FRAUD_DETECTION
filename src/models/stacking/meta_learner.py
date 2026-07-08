from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from loguru import logger


# ── Decision thresholds ───────────────────────────────────────────────────────
AUTO_APPROVE_THRESHOLD = 0.20
BLOCK_THRESHOLD        = 0.70

STACKING_INPUT_COLS = [
    "xgb_prob",
    "lgbm_prob",
    "catboost_prob",
]



def assemble_meta_features(
    probs: Dict[str, np.ndarray],
    required_cols: List[str] = STACKING_INPUT_COLS,
) -> np.ndarray:
    """
    Build (N, n_models) array from dict of model probability arrays.
    If a model's output is missing, fills with the mean of available models.
    """
    n = len(next(iter(probs.values())))
    arrays = []

    available_probs = np.vstack([v for v in probs.values()]).T  # (N, n_available)
    fallback        = available_probs.mean(axis=1)               # mean across available

    for col in required_cols:
        if col in probs:
            arrays.append(probs[col].reshape(-1, 1))
        else:
            logger.warning(
                "Stacking: '{}' not in probs dict — filling with mean of available models.", col
            )
            arrays.append(fallback.reshape(-1, 1))

    return np.hstack(arrays)   # (N, n_models)


def train_meta_learner(
    train_probs:  Dict[str, np.ndarray],
    y_train:      np.ndarray,
    val_probs:    Dict[str, np.ndarray],
    y_val:        np.ndarray,
    save_dir:     str = "artifacts/checkpoints",
) -> dict:
    """
    Train Logistic Regression meta-learner on base model OOF predictions.

    Parameters
    ----------
    train_probs : {model_name: oof_probability_array} — OOF preds on train set
    y_train     : training labels
    val_probs   : {model_name: val_probability_array}
    y_val       : validation labels

    Returns
    -------
    dict with 'model', 'scaler', 'val_preds', 'metrics'
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    X_train = assemble_meta_features(train_probs)
    X_val   = assemble_meta_features(val_probs)

    logger.info(
        "Stacking meta-learner: {} base models, {} train samples, {} val samples.",
        X_train.shape[1], len(X_train), len(X_val),
    )

    # Scale inputs (LogReg is sensitive to scale)
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # Raw LogReg — NO calibration here.
    # The stacking_score must remain a smooth, continuous signal for the BiLSTM.
    # Calibration (isotonic) would bin the probabilities into flat step-wise blocks,
    # destroying gradient information the BiLSTM needs.
    meta_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    meta_model.fit(X_train_s, y_train)

    val_preds = meta_model.predict_proba(X_val_s)[:, 1]

    val_auc  = roc_auc_score(y_val, val_preds)
    val_aupr = average_precision_score(y_val, val_preds)
    logger.info(
        "Stacking meta-learner — Val AUC-ROC: {:.5f}  |  AUC-PR: {:.5f}",
        val_auc, val_aupr,
    )

    # Log individual model contributions
    logger.info("Base model performance on val:")
    for col in STACKING_INPUT_COLS:
        if col in val_probs:
            m_auc  = roc_auc_score(y_val, val_probs[col])
            m_aupr = average_precision_score(y_val, val_probs[col])
            logger.info("  {:25s}  AUC-ROC: {:.5f}  AUC-PR: {:.5f}", col, m_auc, m_aupr)

    # Save
    with open(save_path / "meta_learner.pkl", "wb") as f:
        pickle.dump({"model": meta_model, "scaler": scaler}, f)
    np.save(save_path / "meta_val_preds.npy", val_preds)
    logger.info("Meta-learner saved to {}", save_path / "meta_learner.pkl")

    return {
        "model":     meta_model,
        "scaler":    scaler,
        "val_preds": val_preds,
        "metrics": {
            "val_auc_roc": val_auc,
            "val_auc_pr":  val_aupr,
        },
    }


def predict_meta_learner(
    meta_model,
    scaler: StandardScaler,
    probs:  Dict[str, np.ndarray],
) -> np.ndarray:
    """Run stacking inference — returns calibrated fraud probabilities."""
    X = assemble_meta_features(probs)
    X_s = scaler.transform(X)
    return meta_model.predict_proba(X_s)[:, 1]


def apply_decision_thresholds(
    probs:           np.ndarray,
    auto_approve_t:  float = AUTO_APPROVE_THRESHOLD,
    block_t:         float = BLOCK_THRESHOLD,
) -> pd.DataFrame:
    """
    Apply 3-tier decision rule.

    Returns DataFrame with columns: fraud_prob, decision, action.
    """
    decisions = []
    for p in probs:
        if p < auto_approve_t:
            decision = "auto_approve"
            action   = "Allow transaction"
        elif p <= block_t:
            decision = "flag_review"
            action   = "Flag for manual review / trigger 3DS"
        else:
            decision = "block"
            action   = "Block + alert fraud team"
        decisions.append({"fraud_prob": float(p), "decision": decision, "action": action})

    df = pd.DataFrame(decisions)
    logger.info(
        "Decision distribution — approve: {:.1%}  review: {:.1%}  block: {:.1%}",
        (df["decision"] == "auto_approve").mean(),
        (df["decision"] == "flag_review").mean(),
        (df["decision"] == "block").mean(),
    )
    return df
