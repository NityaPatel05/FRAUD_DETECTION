"""The four closed-loop experiments. Each ``run_arm`` simulates one world with one learning policy in the loop and
returns outcome summaries (ground truth used only for scoring) plus snapshot metrics on the fixed evaluation stream."""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pandas as pd

from src.novelty.attack_detection import add_amount_band, detect_attacks
from src.novelty.drift import psi
from src.simulator import simulate

from .actions import CostRule, SegmentThresholdRule, ThresholdRule
from .config import COST_DEFAULTS, DRIFT, MODEL, MODEL_QUICK, POLICY, POLICY_QUICK, make_world_cfg
from .evaluate import build_eval_stream, metrics, score_snapshots
from .outcomes import add_cost_columns, by_period, summarize
from .policy import ModelPolicy
from .triggers import CalendarTrigger, DriftTrigger, NeverTrigger

PERIOD = 14
NO_EXPLORE = 1e-9


class Setup:
    """Everything that depends on the run size."""

    def __init__(self, size: str, seeds: List[int], eval_seed: int = 777, threshold: float | None = None):
        self.size, self.seeds, self.eval_seed = size, seeds, eval_seed
        self.pol = dict(POLICY_QUICK if size == "quick" else POLICY)
        if threshold is not None:
            self.pol["decline_threshold"] = threshold
        self.model = dict(MODEL_QUICK if size == "quick" else MODEL)
        self.cfg0 = make_world_cfg(0, size)
        self.eval_df: pd.DataFrame | None = None

    def world(self, seed: int, **over):
        return make_world_cfg(seed, self.size, **over)

    def eval_stream(self) -> pd.DataFrame:
        if self.eval_df is None:
            self.eval_df = build_eval_stream(self.world(self.eval_seed, explore_rate=0.0))
        return self.eval_df

    @property
    def fixed_days(self):
        return (self.pol["warmup_days"], self.cfg0.n_days - self.cfg0.test_days)

    def policy(self, name, rule, trigger=None, seed=0, **kw):
        return ModelPolicy(name, rule, trigger or CalendarTrigger(self.pol["retrain_every"]), warmup_days=self.pol["warmup_days"],
                           gap_days=self.pol["gap_days"], neg_keep=self.pol["neg_keep"], max_weight=self.pol["max_weight"],
                           model_params=self.model, seed=seed, **kw)


def run_arm(S: Setup, name: str, seed: int, policy: ModelPolicy, explore: float = 0.0, label_mode: str = "realistic",
            keep_tx: bool = False, cost: Dict | None = None) -> Dict:
    t0 = time.time()
    # eps = 1e-9 instead of 0 for "no exploration": the engine then draws the same random numbers in every arm, so arms
    # sharing a seed see the SAME world until their decisions first differ (paired comparison, same RNG stream)
    cfg = S.world(seed, explore_rate=max(explore, NO_EXPLORE), label_mode=label_mode)
    res = simulate(cfg, policy=policy)
    tx = res.transactions
    o = add_cost_columns(tx, cfg.start_date, cost, explore)
    horizon = cfg.n_days - cfg.test_days
    pre = o[o["day"] < horizon]
    snap = score_snapshots(policy, S.eval_stream(), S.fixed_days, PERIOD, cfg.n_days)
    out = dict(arm=name, seed=seed, sec=round(time.time() - t0, 1), n_retrains=len(policy.retrains),
               retrains=[{k: v for k, v in r.items()} for r in policy.retrains],
               summary_post_warmup=summarize(o[(o["day"] >= S.pol["warmup_days"]) & (o["day"] < horizon)]),
               summary_all_days=summarize(o), summary_test_period=summarize(o[o["day"] >= horizon]),
               series=by_period(o, PERIOD, cfg.n_days), snapshots=snap["snapshots"], aligned=snap["aligned"],
               bust_out=_bust_out_live(o, tx, cfg))
    if hasattr(policy.trigger, "log"):
        out["trigger_log"] = policy.trigger.log[-30:]
    if keep_tx:
        out["_tx"], out["_policy"], out["_cfg"] = tx, policy, cfg
    return out


def _bust_out_live(o: pd.DataFrame, tx: pd.DataFrame, cfg) -> Dict:
    b = o[o["scenario"] == "bust_out"]
    return dict(rows=int(len(b)), approved=int(b["approved_fraud"].sum()), blocked=int(b["blocked_fraud"].sum()),
                loss_usd=float(b["fraud_loss"].sum()))


# ───────────────────────── Experiment 1 ─────────────────────────
def exp1_selection_bias(S: Setup, keep_tx: bool = False) -> Dict:
    thr = S.pol["decline_threshold"]
    specs = [("no_explore", 0.0, True, "realistic", False), ("explore_2pct", 0.02, True, "realistic", False),
             ("explore_5pct", 0.05, True, "realistic", False), ("explore_2pct_no_ipw", 0.02, False, "realistic", False),
             ("oracle_labels", 0.0, True, "oracle", True)]
    arms = []
    for name, eps, ipw, lm, on_all in specs:
        for seed in S.seeds:
            pol = S.policy(name, ThresholdRule(thr), seed=seed, use_ipw=ipw, train_on_all=on_all)
            arms.append(run_arm(S, name, seed, pol, explore=eps, label_mode=lm, keep_tx=keep_tx and name == "explore_2pct"))
    return dict(arms=arms, threshold=thr)


# ───────────────────────── Experiment 2 ─────────────────────────
def exp2_retraining(S: Setup, reuse_calendar14: List[Dict] | None = None) -> Dict:
    thr, eps = S.pol["decline_threshold"], 0.02
    arms = []
    for name, trig in [("no_retrain", NeverTrigger()), ("calendar_30d", CalendarTrigger(30)),
                       ("drift_triggered", DriftTrigger(**{k: v for k, v in DRIFT.items()}))]:
        for seed in S.seeds:
            if name == "drift_triggered":
                trig = DriftTrigger(**DRIFT)
            pol = S.policy(name, ThresholdRule(thr), trigger=trig, seed=seed)
            arms.append(run_arm(S, name, seed, pol, explore=eps, keep_tx=True))
    if reuse_calendar14:
        for a in reuse_calendar14:
            b = dict(a)
            b["arm"] = "calendar_14d"
            arms.append(b)
    return dict(arms=arms, threshold=thr, explore=eps)


# ───────────────────────── Experiment 3 ─────────────────────────
def exp3_cost(S: Setup, cost: Dict | None = None) -> Dict:
    cost = dict(COST_DEFAULTS, **(cost or {}))
    eps = 0.02
    arms = []
    for name in ["single_threshold", "segment_thresholds", "expected_cost"]:
        for seed in S.seeds:
            if name == "expected_cost":
                pol = S.policy(name, CostRule(cost), seed=seed, cost=cost)
            else:
                pol = S.policy(name, SegmentThresholdRule({}, S.pol["decline_threshold"]), seed=seed, cost=cost,
                               tune="segment" if name == "segment_thresholds" else "global")
            arms.append(run_arm(S, name, seed, pol, explore=eps, cost=cost))
    return dict(arms=arms, cost=cost, explore=eps)


# ───────────────────────── Experiment 4 ─────────────────────────
NOV_SEGMENTS = ["mcc", "amount_band", "channel", "merchant_country", "ip_type"]
NOV = dict(period_sec=86400, z_threshold=6.0, min_periods=1, min_volume=5, baseline_periods=28)


def exp4_novelty(S: Setup, arms_tx: List[Dict], ev: pd.DataFrame | None = None) -> Dict:
    """``arms_tx`` entries carry _tx / _policy / _cfg (kept by run_arm). Novelty layer: daily segment-burst alerts
    (src.novelty.attack_detection) + score / feature PSI drift (src.novelty.drift) vs the supervised model."""
    out = []
    for a in arms_tx:
        tx, pol, cfg = a["_tx"], a["_policy"], a["_cfg"]
        d = tx[tx["txn_type"] != "refund"].copy()
        start = pd.Timestamp(cfg.start_date, tz="UTC")
        d["day"] = ((d["event_time"] - start).dt.total_seconds() // 86400).astype(int)
        d["TransactionDT"] = d["day"] * 86400 + ((d["event_time"] - start).dt.total_seconds() % 86400).astype(int)
        d["TransactionAmt"] = pd.to_numeric(d["amount_usd"], errors="coerce").astype("float64")
        d["amount_band"] = add_amount_band(d)
        for c in ("mcc", "merchant_country", "ip_type", "channel"):
            d[c] = d[c].astype(str)
        test0 = cfg.n_days - cfg.test_days
        alerts = detect_attacks(d, NOV_SEGMENTS, NOV["period_sec"], NOV["z_threshold"], NOV["min_periods"],
                                NOV["min_volume"], NOV["baseline_periods"])
        alerts["start_day"] = alerts["start_time"] // 86400
        alerts["end_day"] = (alerts["end_time"] - 1) // 86400
        bo = d[d["fraud_scenario"] == "bust_out"]
        incidents = []
        for iid, g in bo.groupby("incident_id"):
            i0, i1 = int(g["day"].min()), int(g["day"].max())
            hit = []
            for _, al in alerts.iterrows():
                if al["end_day"] < i0 or al["start_day"] > i1 + 1:
                    continue
                m = (g["day"] >= al["start_day"]) & (g["day"] <= al["end_day"]) & (g[al["segment"]].astype(str) == al["value"])
                if m.any():
                    hit.append(al)
            first = min((int(h["start_day"]) for h in hit), default=None)
            incidents.append(dict(incident=int(iid), rows=int(len(g)), first_day=i0, last_day=i1, detected=bool(hit),
                                  detection_day=first, delay_days=None if first is None else max(0, first - i0),
                                  n_alerts=len(hit)))
        pre_alerts = alerts[alerts["start_day"] < test0]
        post_alerts = alerts[alerts["start_day"] >= test0]
        # alerts that cover no bust-out row (false alarms in the test period)
        bo_days = set(zip(bo["day"], bo["mcc"], bo["amount_band"], bo["channel"], bo["merchant_country"], bo["ip_type"]))
        fa_post = 0
        for _, al in post_alerts.iterrows():
            m = bo[(bo["day"] >= al["start_day"]) & (bo["day"] <= al["end_day"]) & (bo[al["segment"]].astype(str) == al["value"])]
            fa_post += int(len(m) == 0)
        # score drift: model score PSI, last model's reference = its first 14 days
        mon = pol.store
        drift = []
        if pol.snapshots:
            last = pol.snapshots[-1]["day"]
            ref = mon.scores(last, last + 14)
            for w0 in range(last + 14, cfg.n_days - 6, 7):
                cur = mon.scores(w0, w0 + 7)
                drift.append(dict(day0=w0, score_psi=float(psi(ref, cur)) if len(cur) > 50 and len(ref) > 50 else None))
        # supervised model on the fixed eval stream (bust_out is present in its test period)
        sup = {}
        if ev is not None and pol.snapshots:
            X = pol.fz.transform(ev)
            p = pol.snapshots[-1]["model"].predict_proba(X)[:, 1]
            day = ev["day"].to_numpy()
            tp = day >= test0
            y = ev["is_fraud_true"].to_numpy()
            bo_m = ev["fraud_scenario"].to_numpy() == "bust_out"
            legit = tp & (y == 0)
            thr = float(np.quantile(p[legit], 0.99)) if legit.any() else 1.0
            other = tp & (y == 1) & ~bo_m
            sup = dict(model_day=pol.snapshots[-1]["day"], fpr_1pct_threshold=thr, bust_out_rows=int((tp & bo_m).sum()),
                       bust_out_recall_at_1pct_fpr=float((p[tp & bo_m] >= thr).mean()) if (tp & bo_m).any() else None,
                       other_fraud_recall_at_1pct_fpr=float((p[other] >= thr).mean()) if other.any() else None)
        out.append(dict(arm=a["arm"], seed=a["seed"], test_start_day=int(test0), incidents=incidents,
                        n_incidents=len(incidents), n_detected=int(sum(i["detected"] for i in incidents)),
                        alerts_pre_test=int(len(pre_alerts)), alerts_in_test=int(len(post_alerts)),
                        false_alarms_in_test_period=int(fa_post), pre_test_days=int(test0),
                        false_alarm_rate_per_week_pre_test=float(len(pre_alerts) / max(test0 - NOV["baseline_periods"], 1) * 7),
                        top_alerts=post_alerts.head(8)[["segment", "value", "start_day", "end_day", "peak_z", "volume"]].to_dict("records"),
                        score_drift=drift, supervised=sup, live=a.get("bust_out")))
    return dict(runs=out, novelty_config=dict(NOV, segments=NOV_SEGMENTS))
