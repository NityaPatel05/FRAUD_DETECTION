"""Evaluation for the simulated world on BOTH truths + slices only a simulator can give."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.training.evaluation import score_report, threshold_at_fpr

SLICE_FPR = 0.01


def _client(df: pd.DataFrame) -> np.ndarray:
    return df["card_id"].fillna(-1).to_numpy().astype(np.int64)


def _day(df: pd.DataFrame) -> np.ndarray:
    return ((df["event_time"] - df["event_time"].min()).dt.total_seconds() // 86400).to_numpy().astype(int)


def _report(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, k: int) -> Dict:
    return score_report(y, p, client=_client(df), amount=df["m_amount_usd"].to_numpy(), day=_day(df), k=k,
                        fpr_targets=(0.005, 0.01))


def sim_report(df: pd.DataFrame, p: np.ndarray, cp_k: int = 100, fpr: float = SLICE_FPR,
               novel=("bust_out",)) -> Dict:
    """Headline metrics on is_fraud_true (real performance) and on reported_label (what the business measures:
    only rows that have an outcome), plus slices at the operating point = 1% FPR on true legit rows."""
    p = np.asarray(p, dtype=np.float64)
    y_true = df["is_fraud_true"].to_numpy().astype(int)
    rep: Dict = {"true": _report(df, y_true, p, cp_k)}
    obs = df["reported_label"].notna().to_numpy()
    if obs.sum() and df.loc[obs, "reported_label"].sum() > 0:
        d = df[obs].reset_index(drop=True)
        rep["reported"] = _report(d, d["reported_label"].to_numpy().astype(int), p[obs], cp_k)
    th = threshold_at_fpr(y_true, p, fpr)
    flagged = p >= th
    rep["operating_point"] = {"fpr_target": fpr, "threshold": th, "flag_rate": float(flagged.mean())}
    sc = {}
    for name, g in df.groupby("fraud_scenario"):
        if name == "legit":
            continue
        m = (df["fraud_scenario"] == name).to_numpy()
        sc[name] = {"n": int(m.sum()), "n_true_fraud": int(y_true[m].sum()), "flagged_rate": float(flagged[m].mean())}
    rep["by_scenario"] = sc
    rep["novel"] = {s: sc.get(s) for s in novel}
    ring = df["ring_id"].notna().to_numpy()
    if ring.any():
        r = pd.DataFrame({"ring": df.loc[ring, "ring_id"].to_numpy(), "hit": flagged[ring]}).groupby("ring")["hit"].max()
        rep["rings"] = {"n_rings": int(len(r)), "rings_detected": float(r.mean())}

    def sl(mask, key):
        mask = np.asarray(mask)
        if mask.sum() > 50 and y_true[mask].sum() > 0:
            r = score_report(y_true[mask], p[mask], amount=df["m_amount_usd"].to_numpy()[mask])
            rep.setdefault("slices", {})[key] = {k: r.get(k) for k in
                                                 ("n", "n_fraud", "base_rate", "auc_pr", "recall_at_fpr_1pct")}
            rep["slices"][key]["recall_at_op"] = float(flagged[mask & (y_true == 1)].mean())
    sl(df["m_account_age_days"].fillna(9999).to_numpy() < 30, "new_customers_lt30d")
    ch = df["m_channel"].astype(str).to_numpy()
    p2p = df["counterparty_customer_id"].notna().to_numpy() if "counterparty_customer_id" in df else np.zeros(len(df), bool)
    sl((ch == "CP"), "card_present")
    sl((ch == "CNP") & ~p2p, "card_not_present")
    sl(p2p, "p2p")
    return rep


def pct_vs_reference(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Percentile of x inside a reference score distribution (e.g. validation scores)."""
    r = np.sort(np.asarray(ref, dtype=np.float64))
    return np.searchsorted(r, x, side="right") / max(len(r), 1)


def recall_for(df: pd.DataFrame, p: np.ndarray, scenario: str, fprs=(0.005, 0.01)) -> Dict:
    y = df["is_fraud_true"].to_numpy().astype(int)
    m = (df["fraud_scenario"] == scenario).to_numpy()
    out = {"n": int(m.sum())}
    for f in fprs:
        th = threshold_at_fpr(y, p, f)
        out[f"recall_at_fpr_{f * 100:g}pct"] = float((p[m] >= th).mean()) if m.any() else float("nan")
    return out
