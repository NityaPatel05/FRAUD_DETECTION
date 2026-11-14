"""
Label-free drift monitoring (Feedzai SAMM / Feature Investigation style).

Chargebacks arrive weeks late, so production teams watch the INPUTS and the SCORE
distribution continuously:
  * PSI per feature and for the model score (common rule of thumb: < 0.1 stable,
    0.1–0.2 moderate, > 0.2 investigate — a convention, not a validated threshold)
  * a domain classifier: if a model can tell "recent" rows from "reference" rows,
    something changed, and its importances say what.
Drift is a reason to look, not proof of fraud (seasonality / promotions drift too).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index with quantile bins from the reference; NaN = own bin."""
    ref = np.asarray(reference, dtype=float); cur = np.asarray(current, dtype=float)
    ref_nan, cur_nan = np.isnan(ref).mean(), np.isnan(cur).mean()
    r, c = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(r) == 0 or len(c) == 0:
        return float("nan")
    edges = np.unique(np.quantile(r, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        p = np.array([1.0 - ref_nan, ref_nan]); q = np.array([1.0 - cur_nan, cur_nan])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(r, edges)[0] / len(ref)
        q = np.histogram(c, edges)[0] / len(cur)
        p = np.append(p, ref_nan); q = np.append(q, cur_nan)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_report(ref_df: pd.DataFrame, cur_df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(ref_df[c].to_numpy(), cur_df[c].to_numpy())}
            for c in cols if c in ref_df.columns and c in cur_df.columns]
    rep = pd.DataFrame(rows).sort_values("psi", ascending=False)
    rep["level"] = pd.cut(rep["psi"], [-np.inf, 0.1, 0.2, np.inf], labels=["stable", "moderate", "investigate"])
    return rep.reset_index(drop=True)


def domain_classifier(ref_df: pd.DataFrame, cur_df: pd.DataFrame, cols: Sequence[str],
                      seed: int = 42) -> Dict:
    """Train ref-vs-current classifier (holdout AUC). AUC ≈ 0.5 → no detectable shift."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    X = pd.concat([ref_df[list(cols)], cur_df[list(cols)]], ignore_index=True)
    y = np.r_[np.zeros(len(ref_df)), np.ones(len(cur_df))]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=seed)
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    imp = pd.Series(m.booster_.feature_importance("gain"), index=list(cols)).sort_values(ascending=False)
    return {"auc": float(auc), "top_drivers": imp.head(15).round(1).to_dict()}
