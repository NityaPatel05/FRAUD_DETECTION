"""
Validation design, metrics, blending and post-processing for the v2 trainers.

Best practices implemented here:
  * time-ordered folds with a gap (label-delay realism), several folds → mean ± std
  * selection metric = average precision (AUC-PR), never ROC-AUC alone
  * operational metrics: precision@top-k%, recall at fixed FPR, client-level precision@k
  * rank-average blending (robust to differently-calibrated models) instead of a
    stacker trained on leaky / partially-zero OOF predictions
  * per-client score smoothing (causal: running mean of the client's scores so far)
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


# ─────────────────────────────────────────────────────────────────────────────
# Folds
# ─────────────────────────────────────────────────────────────────────────────

def time_folds(day: np.ndarray, n_folds: int = 3, val_days: int = 15,
               gap_days: int = 7) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Walk-forward folds over the LAST n_folds*val_days days of ``day``.

    fold k: train = days < val_start_k - gap ; val = [val_start_k, val_start_k + val_days)
    """
    day = np.asarray(day)
    end = int(day.max()) + 1
    folds = []
    for k in range(n_folds):
        v_start = end - (n_folds - k) * val_days
        v_end = v_start + val_days
        tr = np.where(day < v_start - gap_days)[0]
        va = np.where((day >= v_start) & (day < v_end))[0]
        if len(tr) and len(va):
            folds.append((tr, va))
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_top(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    k = max(1, int(round(len(p) * frac)))
    top = np.argsort(-p, kind="mergesort")[:k]
    return float(np.mean(y[top]))


def recall_at_fpr(y: np.ndarray, p: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr[fpr <= fpr_target])) if np.any(fpr <= fpr_target) else 0.0


def client_precision_at_k(y: np.ndarray, p: np.ndarray, client: np.ndarray, k: int) -> float:
    """Investigators review clients, not rows: rank clients by their max score,
    a client is a hit if any of its rows is fraud (Fraud Detection Handbook CP@k)."""
    df = pd.DataFrame({"c": client, "y": y, "p": p})
    df = df[df["c"] >= 0]
    g = df.groupby("c").agg(p=("p", "max"), y=("y", "max"))
    top = g.sort_values("p", ascending=False).head(k)
    return float(top["y"].mean()) if len(top) else float("nan")


def score_report(y: np.ndarray, p: np.ndarray, client: np.ndarray | None = None) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    rep = {
        "auc_pr": float(average_precision_score(y, p)),
        "auc_roc": float(roc_auc_score(y, p)),
        "base_rate": float(y.mean()),
        "precision_top_0.5pct": precision_at_top(y, p, 0.005),
        "precision_top_1pct": precision_at_top(y, p, 0.01),
        "recall_at_fpr_1pct": recall_at_fpr(y, p, 0.01),
        "recall_at_fpr_5pct": recall_at_fpr(y, p, 0.05),
    }
    if client is not None:
        rep["client_precision_at_500"] = client_precision_at_k(y, p, np.asarray(client), 500)
    return rep


# ─────────────────────────────────────────────────────────────────────────────
# Blending
# ─────────────────────────────────────────────────────────────────────────────

def rank_norm(p: np.ndarray) -> np.ndarray:
    return (rankdata(p) - 1) / max(len(p) - 1, 1)


def rank_blend(preds: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    tot = sum(weights.values())
    return sum(weights[k] * rank_norm(preds[k]) for k in weights) / tot


def search_blend_weights(oof: Dict[str, np.ndarray], y: np.ndarray,
                         fold_ids: np.ndarray, step: float = 0.1) -> Dict[str, float]:
    """Grid-search simplex weights maximising the MEAN per-fold AP of the rank
    blend (ranks computed within each fold, as each fold is a separate period)."""
    names = list(oof)
    if len(names) == 1:
        return {names[0]: 1.0}
    grid = np.round(np.arange(0, 1 + 1e-9, step), 6)
    best, best_w = -1.0, None
    for w in product(grid, repeat=len(names)):
        if abs(sum(w) - 1) > 1e-6:
            continue
        aps = []
        for f in np.unique(fold_ids):
            m = fold_ids == f
            blend = sum(wi * rank_norm(oof[n][m]) for wi, n in zip(w, names))
            aps.append(average_precision_score(y[m], blend))
        if np.mean(aps) > best:
            best, best_w = float(np.mean(aps)), w
    return {n: float(wi) for n, wi in zip(names, best_w)}


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def client_smoothing(p: np.ndarray, client: np.ndarray, t: np.ndarray,
                     mode: str = "causal", alpha: float = 0.5) -> np.ndarray:
    """Blend each score with the client's mean score.

    causal: running mean over the client's rows so far (servable).
    transductive: mean over all the client's rows (Kaggle 1st-place style).
    Rows without a client id keep their raw score.
    """
    df = pd.DataFrame({"c": client, "p": p, "t": t, "i": np.arange(len(p))})
    df = df.sort_values(["c", "t", "i"], kind="mergesort")
    g = df.groupby("c")["p"]
    m = g.cumsum() / (g.cumcount() + 1) if mode == "causal" else g.transform("mean")
    df["m"] = m
    df = df.sort_values("i")
    out = (1 - alpha) * p + alpha * df["m"].to_numpy()
    return np.where(np.asarray(client) >= 0, out, p)
