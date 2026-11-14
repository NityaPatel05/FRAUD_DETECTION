"""
Phase 8 — champion/challenger promotion gate on shadow traffic.

  python scripts/run_promotion_gate.py --labels data/processed/val_raw.parquet [--promote]

Reads the decision log (which contains the challenger's shadow score when the service
was started with a challenger), applies the SAME cost policy to the challenger's scores,
and runs src/feedback/promotion.promotion_gate on matured labels. With --promote and a
passing gate, the model registry's champion is replaced by the challenger.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import pandas as pd
import yaml

from src.feedback.monitoring import load_decision_log
from src.feedback.promotion import promotion_gate
from src.jobs.retrain_v2 import ModelRegistry
from src.serving.decision import CostPolicy


def main(args):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    log = load_decision_log(cfg["decision_v2"]["decision_log"])
    if "challenger.p_fraud" not in log.columns:
        raise SystemExit("No challenger scores in the decision log — start the service with a challenger first.")
    log = log[log["decided_by"] == "policy"]                          # compare model-driven decisions only
    lab = pd.read_parquet(args.labels, columns=["TransactionID", "isFraud"]).set_index("TransactionID")["isFraud"]
    y = log["transaction_id"].map(lab).to_numpy(dtype=float)
    pol = CostPolicy(cfg["decision_v2"])
    amt = log["amount"].to_numpy(dtype=float)
    act_ch = np.array([pol.decide(p, a)["action"] for p, a in zip(log["challenger.p_fraud"], amt)])
    res = promotion_gate(y, log["p_fraud"].to_numpy(), log["challenger.p_fraud"].to_numpy(),
                         log["action"].to_numpy(), act_ch, **cfg["feedback_v2"]["promotion"])
    print(json.dumps(res, indent=1, default=str))
    if args.promote and res["promote"]:
        reg = ModelRegistry(cfg["feedback_v2"]["registry"])
        reg.promote_challenger(evidence=res)
        print("Challenger promoted to champion.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--promote", action="store_true")
    main(ap.parse_args())
