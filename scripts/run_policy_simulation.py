"""
Phase 7 — decision-policy simulation on the val holdout (no training).

  python scripts/run_policy_simulation.py [--schema raw]

Uses artifacts/v2/<tag>/val_predictions.parquet (written by run_training.py):
  1. sets review_min_p so the review queue fits review_capacity_per_day
  2. applies the cost-based policy to every val transaction
  3. reports money: realised cost vs "approve everything" and vs a plain score threshold,
     fraud $ stopped, decline / step-up / review rates, precision of declines
Writes policy_report.json and policy_overrides.json (read by the decision service).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.feature_engineering.pipeline_v2 import dataset_tag
from src.serving.decision import CostPolicy


def realised_cost(action: np.ndarray, y: np.ndarray, amt: np.ndarray, c: dict) -> np.ndarray:
    loss = amt * c["fraud_loss_rate"] + c["chargeback_fee"]
    legit = amt * c["margin_rate"]
    cost = np.zeros(len(y))
    a = action
    cost[(a == "approve") & (y == 1)] = loss[(a == "approve") & (y == 1)]
    m = a == "decline"
    cost[m] = np.where(y[m] == 1, c["decline_cost"],
                       legit[m] + c.get("false_decline_churn_cost", 0.0) + c["decline_cost"])
    m = a == "step_up"
    cost[m] = c["step_up_cost"] + np.where(y[m] == 1, loss[m] * (1 - c["step_up_fraud_stop_rate"]),
                                           legit[m] * c["step_up_legit_abandon_rate"])
    m = a == "review"
    cost[m] = c["review_cost"] + np.where(y[m] == 1, loss[m] * (1 - c["review_fraud_stop_rate"]), 0.0)
    return cost


def apply_policy(p, amt, cfg):
    pol = CostPolicy(cfg)
    return np.array([pol.decide(pi, ai)["action"] for pi, ai in zip(p, amt)])


def main(args):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    dcfg = dict(cfg["decision_v2"])
    mode = args.agg_mode or cfg["fe_v2"]["agg_mode"]
    tag = dataset_tag(mode, args.schema or cfg["fe_v2"].get("schema", "full"))
    out = Path(cfg["paths"]["artifacts_v2"].format(mode=tag))
    vp = pd.read_parquet(out / "val_predictions.parquet")
    p = vp["p_blend"].to_numpy(); y = vp["isFraud"].to_numpy().astype(int)
    amt = vp["TransactionAmt"].to_numpy(dtype=float)
    days = vp["TransactionDT"].to_numpy() // 86400
    n_days = len(np.unique(days))

    # 1. review threshold from capacity
    dcfg["review_min_p"] = 0.0
    act = apply_policy(p, amt, dcfg)
    per_day = pd.Series(act == "review").groupby(days).sum()
    review_min_p = 0.0
    if per_day.max() > dcfg["review_capacity_per_day"]:
        cand = np.sort(p[act == "review"])[::-1]
        for th in np.quantile(cand, np.linspace(0, 1, 201)):          # smallest threshold that fits
            if pd.Series((act == "review") & (p >= th)).groupby(days).sum().max() <= dcfg["review_capacity_per_day"]:
                review_min_p = float(th); break
    dcfg["review_min_p"] = review_min_p
    act = apply_policy(p, amt, dcfg)

    # 2-3. money
    cost = realised_cost(act, y, amt, dcfg)
    approve_all = realised_cost(np.full(len(y), "approve"), y, amt, dcfg)
    # baseline: decline the top-k by score with the same decline volume
    k = int((act == "decline").sum())
    thr_act = np.full(len(y), "approve", dtype=object)
    if k:
        thr_act[np.argsort(-p)[:k]] = "decline"
    thr_cost = realised_cost(thr_act.astype(str), y, amt, dcfg)
    stopped = np.isin(act, ["decline", "review", "step_up"]) & (y == 1)
    rep = {
        "n_transactions": int(len(y)), "days": int(n_days), "review_min_p": review_min_p,
        "action_rates": pd.Series(act).value_counts(normalize=True).round(4).to_dict(),
        "reviews_per_day_max": int(pd.Series(act == "review").groupby(days).sum().max()),
        "cost_policy": float(cost.sum()), "cost_approve_all": float(approve_all.sum()),
        "cost_score_threshold_same_declines": float(thr_cost.sum()),
        "savings_vs_approve_all": float(approve_all.sum() - cost.sum()),
        "fraud_amount_total": float(amt[y == 1].sum()),
        "fraud_amount_intercepted": float(amt[stopped].sum()),
        "decline_precision": float(y[act == "decline"].mean()) if (act == "decline").any() else None,
        "fraud_recall_any_action": float(stopped.sum() / max(y.sum(), 1)),
        "cost_assumptions": {k_: v for k_, v in dcfg.items() if isinstance(v, (int, float))},
    }
    (out / "policy_report.json").write_text(json.dumps(rep, indent=1))
    (out / "policy_overrides.json").write_text(json.dumps({"review_min_p": review_min_p}, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg-mode", choices=["causal", "transductive"], default=None)
    ap.add_argument("--schema", choices=["full", "raw"], default=None)
    main(ap.parse_args())
