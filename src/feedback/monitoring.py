"""
Production monitoring from the decision log (+ matured labels when available).

Daily: action rates, p99 latency, score-distribution shift (PSI vs a reference window)
and — once labels mature — false-positive rate and recall of flagged transactions.
Alerts fire when a metric leaves its band (e.g. a false-positive-rate drift alert, which
catches "the new model is quietly declining good customers" before support tickets do).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.novelty.drift import psi


def load_decision_log(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.json_normalize(rows)
    df["day"] = (df["event_time"].astype(float) // 86400).astype(int)
    return df


def daily_report(log: pd.DataFrame, labels: Optional[pd.Series] = None, ref_days: int = 7) -> pd.DataFrame:
    """labels: Series indexed by transaction_id (1 / 0 / NaN = immature)."""
    out = []
    days = sorted(log["day"].unique())
    ref = log[log["day"] < days[0] + ref_days]["p_fraud"].to_numpy()
    for d in days:
        g = log[log["day"] == d]
        rec = {"day": int(d), "n": len(g),
               "p99_latency_ms": float(np.percentile(g["latency_ms.total"], 99)),
               "score_psi": psi(ref, g["p_fraud"].to_numpy()) if d >= days[0] + ref_days else np.nan}
        for a in ["approve", "step_up", "review", "decline"]:
            rec[f"rate_{a}"] = float((g["action"] == a).mean())
        if labels is not None:
            y = g["transaction_id"].map(labels)
            flagged = g["action"].isin(["decline", "review"])
            legit, fraud = (y == 0), (y == 1)
            rec["fpr_flagged"] = float((flagged & legit).sum() / legit.sum()) if legit.any() else np.nan
            rec["recall_flagged"] = float((flagged & fraud).sum() / fraud.sum()) if fraud.any() else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def alerts(report: pd.DataFrame, bands: Dict) -> pd.DataFrame:
    """bands: {"rate_decline": [lo, hi], "p99_latency_ms": [0, 50], "score_psi": [0, 0.2], "fpr_flagged": [0, 0.05]}"""
    rows = []
    for _, r in report.iterrows():
        for metric, (lo, hi) in bands.items():
            v = r.get(metric)
            if v is not None and not pd.isna(v) and not (lo <= v <= hi):
                rows.append({"day": int(r["day"]), "metric": metric, "value": float(v), "band": [lo, hi]})
    return pd.DataFrame(rows, columns=["day", "metric", "value", "band"])
