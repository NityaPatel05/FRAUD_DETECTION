"""Retraining triggers: never / calendar / drift-triggered (score PSI, feature PSI, delayed-label performance)."""
from __future__ import annotations

from typing import Dict, Protocol, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score

from src.novelty.drift import psi          # the same PSI the novelty / monitoring layer uses

from .config import DRIFT

MONITORED = ["log_amt", "amt_z_card", "hist_card_n_1h", "hist_card_n_24h", "hist_ip_n_1h", "hist_ip_n_24h",
             "hist_device_n_24h", "hist_merchant_n_1h", "hist_card_decl_7d", "account_age_days",
             "ip_country_mismatch", "ip_type", "avs_result", "cvc_check", "three_ds_result", "mcc"]


class Monitor(Protocol):
    def scores(self, d0: int, d1: int) -> np.ndarray: ...
    def feature(self, name: str, d0: int, d1: int) -> np.ndarray: ...
    def matured_perf(self, d0: int, d1: int) -> Tuple[float, int]: ...


class NeverTrigger:
    name = "no_retrain"

    def should_retrain(self, day: int, last_train_day: int, mon: Monitor) -> Tuple[bool, str]:
        return False, ""


class CalendarTrigger:
    def __init__(self, every: int = 30):
        self.every = every
        self.name = f"calendar_{every}d"

    def should_retrain(self, day: int, last_train_day: int, mon: Monitor) -> Tuple[bool, str]:
        return (day - last_train_day >= self.every), "calendar"


class DriftTrigger:
    """Fire when, comparing the post-deployment reference window with the latest window:
       * the score distribution moved (PSI >= score_psi), or
       * >= feat_psi_k monitored features moved (PSI >= feat_psi), or
       * the model's AUC-PR on newly MATURED labels fell by perf_drop (relative) with enough positives.
    Checks run every ``check_every`` days, never sooner than ``min_gap_days`` after the last retrain."""
    name = "drift_triggered"

    def __init__(self, features: Sequence[str] = MONITORED, **kw):
        self.p = dict(DRIFT, **kw)
        self.features = list(features)
        self.log: list = []

    def evaluate(self, ref_scores, cur_scores, ref_feats: Dict[str, np.ndarray], cur_feats: Dict[str, np.ndarray],
                 perf_ref: Tuple[float, int] | None = None, perf_cur: Tuple[float, int] | None = None) -> Dict:
        p = self.p
        s_psi = psi(ref_scores, cur_scores) if len(ref_scores) > 50 and len(cur_scores) > 50 else 0.0
        f_psi = {f: psi(ref_feats[f], cur_feats[f]) for f in self.features
                 if f in ref_feats and len(ref_feats[f]) > 50 and len(cur_feats.get(f, [])) > 50}
        n_f = sum(v >= p["feat_psi"] for v in f_psi.values())
        perf_drop = False
        if perf_ref and perf_cur and perf_ref[1] >= p["perf_min_pos"] and perf_cur[1] >= p["perf_min_pos"]:
            perf_drop = perf_cur[0] < perf_ref[0] * (1 - p["perf_drop"])
        reasons = []
        if s_psi >= p["score_psi"]:
            reasons.append(f"score_psi={s_psi:.2f}")
        if n_f >= p["feat_psi_k"]:
            reasons.append(f"feat_psi({n_f} feats)")
        if perf_drop:
            reasons.append(f"perf {perf_cur[0]:.2f}<{perf_ref[0]:.2f}")
        return dict(fire=bool(reasons), reasons=reasons, score_psi=float(s_psi), n_feat_drift=int(n_f),
                    max_feat_psi=float(max(f_psi.values())) if f_psi else 0.0,
                    perf_ref=perf_ref, perf_cur=perf_cur)

    def should_retrain(self, day: int, last_train_day: int, mon: Monitor) -> Tuple[bool, str]:
        p = self.p
        if day - last_train_day < max(p["min_gap_days"], p["ref_days"] + p["window_days"]):
            return False, ""
        if (day - last_train_day) % p["check_every"] != 0 and (day - last_train_day) > p["ref_days"] + p["window_days"]:
            return False, ""
        r0, r1 = last_train_day, last_train_day + p["ref_days"]
        c0, c1 = day - p["window_days"], day
        ref_s, cur_s = mon.scores(r0, r1), mon.scores(c0, c1)
        ref_f = {f: mon.feature(f, r0, r1) for f in self.features}
        cur_f = {f: mon.feature(f, c0, c1) for f in self.features}
        gap = p.get("gap_days", 30)
        perf_ref = perf_cur = None
        if day - gap - p["window_days"] >= r1:
            perf_ref = mon.matured_perf(r0, r1) if day - gap >= r1 else None
            perf_cur = mon.matured_perf(day - gap - p["window_days"], day - gap)
        res = self.evaluate(ref_s, cur_s, ref_f, cur_f, perf_ref, perf_cur)
        res["day"] = day
        self.log.append(res)
        return res["fire"], ",".join(res["reasons"])


def auc_pr(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if len(y) and y.sum() > 0 else float("nan")
