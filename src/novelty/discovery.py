"""
Unsupervised discovery for ANALYSTS (not a blocking score).

Production evidence: Isolation Forest / micro-clustering surfaces new behaviour
clusters worth investigating (FRAUDGUESS 2025; Grab GraphBEAN feeds analysts), but
anomalies ≠ fraud, so nobody found in public sources blocks on them. Output here is
a ranked list of unusual CLUSTERS with what they share, for the review queue's
exploration slots.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def discover_unusual_clusters(df: pd.DataFrame, num_cols: Sequence[str], cat_cols: Sequence[str] = (),
                              top_frac: float = 0.01, n_clusters: int = 10, seed: int = 42) -> pd.DataFrame:
    """Isolation Forest → top ``top_frac`` most anomalous rows → KMeans clusters →
    per-cluster size, mean anomaly, dominant categorical values, known fraud rate."""
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    X = df[list(num_cols)].astype(float)
    X = X.fillna(X.median())
    Xs = StandardScaler().fit_transform(np.log1p(X.clip(lower=0)) if (X >= 0).all().all() else X)
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=-1).fit(Xs)
    score = -iso.score_samples(Xs)                                     # higher = more anomalous
    top = np.argsort(-score)[: max(n_clusters, int(len(df) * top_frac))]
    k = min(n_clusters, len(top))
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xs[top])
    sub = df.iloc[top].assign(_cluster=lab, _anomaly=score[top])
    rows = []
    for c, g in sub.groupby("_cluster"):
        rec = {"cluster": int(c), "size": len(g), "mean_anomaly": float(g["_anomaly"].mean())}
        for col in cat_cols:
            if col in g.columns:
                vc = g[col].astype(str).value_counts(normalize=True)
                rec[f"top_{col}"] = f"{vc.index[0]} ({vc.iloc[0]:.0%})"
        if "isFraud" in g.columns and g["isFraud"].notna().any():
            rec["known_fraud_rate"] = float(g["isFraud"].mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("mean_anomaly", ascending=False).reset_index(drop=True)
