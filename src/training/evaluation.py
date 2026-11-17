"""
Validation design, metrics, blending and post-processing for the v2 trainers.

Best practices implemented here:
  * rolling (prequential) time folds: train | gap | val, several folds -> mean +- SE
  * training targets "as known at the cutoff" (src.labels.policy), never the final label
  * early stopping NEVER on the rows that are scored: each fold's val window is split in
    time into an early-stopping part (first half) and a scoring part (second half)
  * selection metric = average precision (AUC-PR), reported next to the base rate
  * operational metrics: recall @ fixed FPR, value detection rate (fraud $ caught / fraud $),
    card/client precision@k per day (Fraud Detection Handbook CP@k), false-decline rate
  * slices: client seen vs cold-start, uid unbuildable, copied positives removed
  * rank-average blending (robust to differently-calibrated models) instead of a
    stacker trained on leaky / partially-zero OOF predictions
  * per-client score smoothing (causal: running mean of the client's scores so far)
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.labels.policy import DAY, LabelPolicy, training_target


# ─────────────────────────────────────────────────────────────────────────────
# Folds
# ─────────────────────────────────────────────────────────────────────────────

def time_folds(day: np.ndarray, n_folds: int = 3, val_days: int = 15,
               gap_days: int = 7) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Walk-forward folds over the LAST n_folds*val_days days of ``day``.

    fold k: train = days < val_start_k - gap ; val = [val_start_k, val_start_k + val_days)
    (legacy index-only API; the trainers use ``make_folds`` which adds targets and ES split)
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


@dataclass
class Fold:
    """One rolling fold. Index arrays refer to rows of the frame the fold was built from."""
    k: int
    train: np.ndarray            # rows t < cutoff - gap
    y_train: np.ndarray          # target AS KNOWN at the cutoff (policy), aligned with ``train``
    es: np.ndarray               # early-stopping rows (first part of the val window, true labels)
    score: np.ndarray            # scoring rows (rest of the val window, true labels, never used to stop)
    cutoff: int                  # seconds: when the model "is deployed" (= val window start)
    val_start_day: int
    val_end_day: int

    @property
    def val(self) -> np.ndarray:
        return np.concatenate([self.es, self.score])


def make_folds(t: np.ndarray, y: np.ndarray, txn_id: np.ndarray, n_folds: int, val_days: int,
               policy: LabelPolicy, gap_days: Optional[float] = None, es_frac: float = 0.5) -> List[Fold]:
    """Rolling folds with policy targets.  train | gap | val(es + score).

    Each val window is ``val_days`` long and the windows tile the last ``n_folds*val_days``
    days. Rows within ``gap_days`` (default policy.train_gap_days) before the cutoff are
    dropped from training; the training target is the label known at the cutoff.
    The first ``es_frac`` of each val window (in time) is used ONLY for early stopping, the
    rest ONLY for scoring.
    """
    t = np.asarray(t, dtype=np.int64)
    day = t // DAY
    gap = policy.train_gap_days if gap_days is None else gap_days
    end = int(day.max()) + 1
    es_days = int(np.clip(round(val_days * es_frac), 0, val_days - 1))
    folds: List[Fold] = []
    for k in range(n_folds):
        v_start = end - (n_folds - k) * val_days
        v_end = v_start + val_days
        cutoff = v_start * DAY
        target, usable = training_target(y, txn_id, t, cutoff, policy, gap)
        tr = np.where(usable)[0]
        es = np.where((day >= v_start) & (day < v_start + es_days))[0]
        sc = np.where((day >= v_start + es_days) & (day < v_end))[0]
        if len(tr) and len(sc):
            folds.append(Fold(k, tr, target[tr].astype(int), es if len(es) else sc, sc, int(cutoff),
                              int(v_start), int(v_end)))
    return folds


def fold_summary(f: Fold, y_true: np.ndarray) -> Dict:
    ys = np.asarray(y_true)[f.score]
    return {"fold": f.k, "val_days": [f.val_start_day, f.val_end_day], "n_train": int(len(f.train)),
            "train_pos_known": int(f.y_train.sum()), "n_es": int(len(f.es)), "n_score": int(len(f.score)),
            "score_pos": int(ys.sum()), "score_base_rate": float(ys.mean())}


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


def threshold_at_fpr(y: np.ndarray, p: np.ndarray, fpr_target: float) -> float:
    """Score threshold (flag if p >= thr) with the largest recall whose FPR <= target; +inf if none."""
    fpr, _, thr = roc_curve(y, p, drop_intermediate=False)     # exact operating point, no corner pruning
    ok = np.where(fpr <= fpr_target)[0]
    return float(thr[ok[-1]]) if len(ok) and np.isfinite(thr[ok[-1]]) else float("inf")


def value_detection_rate(y: np.ndarray, p: np.ndarray, amount: np.ndarray, fpr_target: float) -> float:
    """Fraud money caught / total fraud money when flagging at a fixed false-positive rate."""
    y = np.asarray(y).astype(int)
    amount = np.asarray(amount, dtype=np.float64)
    tot = amount[y == 1].sum()
    if tot <= 0:
        return float("nan")
    thr = threshold_at_fpr(y, p, fpr_target)
    caught = amount[(y == 1) & (np.asarray(p) >= thr)].sum()
    return float(caught / tot)


def client_precision_at_k(y: np.ndarray, p: np.ndarray, client: np.ndarray, k: int) -> float:
    """Clients ranked by their max score over the whole window (legacy single-shot CP@k)."""
    df = pd.DataFrame({"c": client, "y": y, "p": p})
    df = df[df["c"] >= 0]
    g = df.groupby("c").agg(p=("p", "max"), y=("y", "max"))
    top = g.sort_values("p", ascending=False).head(k)
    return float(top["y"].mean()) if len(top) else float("nan")


def card_precision_at_k(y: np.ndarray, p: np.ndarray, client: np.ndarray, day: np.ndarray,
                        k: int = 100) -> Tuple[float, int]:
    """Fraud Detection Handbook CP@k, averaged over days.

    Each day: score every client by its max score that day, investigators review the top
    k clients; precision = share of them with a fraud that day. A client found to be
    compromised is blocked, so it is removed from all later days.
    Returns (mean daily CP@k, number of days used). Rows without a client are skipped.
    """
    df = pd.DataFrame({"c": np.asarray(client), "y": np.asarray(y).astype(int),
                       "p": np.asarray(p, dtype=np.float64), "d": np.asarray(day)})
    df = df[df["c"] >= 0]
    blocked: set = set()
    precisions = []
    for _, g in df.groupby("d", sort=True):
        g = g[~g["c"].isin(blocked)]
        if g.empty:
            continue
        agg = g.groupby("c").agg(p=("p", "max"), y=("y", "max"))
        top = agg.sort_values("p", ascending=False, kind="mergesort").head(k)
        precisions.append(float(top["y"].sum()) / min(k, len(agg)))
        blocked.update(top.index[top["y"] == 1].tolist())
    return (float(np.mean(precisions)) if precisions else float("nan")), len(precisions)


def score_report(y: np.ndarray, p: np.ndarray, client: np.ndarray | None = None,
                 amount: np.ndarray | None = None, day: np.ndarray | None = None,
                 k: int = 100, fpr_targets: Sequence[float] = (0.005, 0.01),
                 review_frac: float = 0.01, threshold: float | None = None) -> Dict[str, float]:
    """All headline metrics for one (labels, scores) set. ``p`` should be a RANKING score.

    amount -> value detection rate;  client+day -> CP@k per day;
    threshold -> false-decline rate at that fixed score threshold (else at the top
    ``review_frac`` of rows, the operating point of a fixed review/decline budget).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    n_pos = int(y.sum())
    rep: Dict[str, float] = {"n": int(len(y)), "n_fraud": n_pos,
                             "base_rate": float(y.mean()) if len(y) else float("nan")}
    if n_pos == 0 or n_pos == len(y):
        return rep
    rep.update({
        "auc_pr": float(average_precision_score(y, p)),
        "auc_roc": float(roc_auc_score(y, p)),
        "precision_top_0.5pct": precision_at_top(y, p, 0.005),
        "precision_top_1pct": precision_at_top(y, p, 0.01),
        "recall_at_fpr_0.5pct": recall_at_fpr(y, p, 0.005),
        "recall_at_fpr_1pct": recall_at_fpr(y, p, 0.01),
        "recall_at_fpr_5pct": recall_at_fpr(y, p, 0.05),
    })
    rep["ap_over_base_rate"] = rep["auc_pr"] / rep["base_rate"]
    if amount is not None:
        for f in fpr_targets:
            rep[f"vdr_at_fpr_{f * 100:g}pct"] = value_detection_rate(y, p, amount, f)
    # false-decline rate: legit rows flagged / legit rows
    if threshold is not None:
        flagged = p >= threshold
        tag = "at_threshold"
    else:
        kk = max(1, int(round(len(p) * review_frac)))
        flagged = np.zeros(len(p), dtype=bool)
        flagged[np.argsort(-p, kind="mergesort")[:kk]] = True
        tag = f"top{review_frac * 100:g}pct"
    legit = y == 0
    rep[f"false_decline_rate_{tag}"] = float((flagged & legit).sum() / max(legit.sum(), 1))
    rep[f"flag_precision_{tag}"] = float(y[flagged].mean()) if flagged.any() else float("nan")
    if client is not None and day is not None:
        cp, nd = card_precision_at_k(y, p, client, day, k)
        rep[f"cp_at_{k}_daily"] = cp
        rep["cp_days"] = nd
    elif client is not None:
        rep[f"client_precision_at_{k}"] = client_precision_at_k(y, p, np.asarray(client), k)
    return rep


# ─────────────────────────────────────────────────────────────────────────────
# Slices (where does the score really come from?)
# ─────────────────────────────────────────────────────────────────────────────

def copied_positive_flags(client: np.ndarray, t: np.ndarray, txn_id: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True for fraud rows that come AFTER the same client's first fraud row.

    IEEE-CIS propagated chargeback labels to the account's later transactions (62.6% of
    fraud rows are such copies): a model "detects" them by remembering the client, which
    a real system would simply have blocked. Rows without a client are never flagged.
    """
    client = np.asarray(client)
    y = np.asarray(y).astype(int)
    order = np.lexsort((np.asarray(txn_id), np.asarray(t)))          # time, then id
    c = client[order]
    yy = y[order]
    prior = pd.Series(yy).groupby(c).cumsum().to_numpy() - yy         # frauds of the client before this row
    flag_sorted = (yy == 1) & (prior > 0) & (c >= 0)
    out = np.zeros(len(y), dtype=bool)
    out[order] = flag_sorted
    return out


def slice_masks(client_score: np.ndarray, client_fit: np.ndarray, copied_score: np.ndarray) -> Dict[str, np.ndarray]:
    """Boolean masks over the scored rows.

    client_seen        client has at least one row in the fit window
    cold_start         client id buildable but never seen in the fit window
    uid_unbuildable    client id could not be built (addr1 / D1 missing)
    copied_removed     everything except fraud rows that are copies of an earlier client fraud
    """
    cs = np.asarray(client_score)
    cf = np.asarray(client_fit)
    known = cs >= 0
    seen = known & np.isin(cs, np.unique(cf[cf >= 0]))
    return {"all": np.ones(len(cs), dtype=bool), "client_seen": seen, "cold_start": known & ~seen,
            "uid_unbuildable": ~known, "copied_removed": ~np.asarray(copied_score, dtype=bool)}


def sliced_report(y: np.ndarray, p: np.ndarray, masks: Dict[str, np.ndarray],
                  amount: np.ndarray | None = None, client: np.ndarray | None = None,
                  day: np.ndarray | None = None, k: int = 100, min_fraud: int = 5) -> Dict[str, Dict]:
    y = np.asarray(y)
    p = np.asarray(p)
    out = {}
    for name, m in masks.items():
        m = np.asarray(m, dtype=bool)
        if m.sum() == 0 or y[m].sum() < min_fraud or y[m].sum() == m.sum():
            out[name] = {"n": int(m.sum()), "n_fraud": int(y[m].sum()), "skipped": "too few fraud rows"}
            continue
        out[name] = score_report(y[m], p[m], None if client is None else np.asarray(client)[m],
                                 None if amount is None else np.asarray(amount)[m],
                                 None if day is None else np.asarray(day)[m], k)
    return out


def mean_se(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and standard error of per-fold values."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


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
