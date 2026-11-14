"""
Phase 8 — why propensity-logged exploration matters (simulation on the val holdout; no training).

  python scripts/run_exploration_simulation.py

Pretend the current policy BLOCKS the riskiest transactions (top `block_rate` by score),
so their labels are never learnt, except for a small explored fraction whose P(allow) is
logged. Then evaluate several CANDIDATE policies three ways:
  truth  — using all labels (only possible in this simulation)
  naive  — using only observed outcomes (what you get without exploration logging)
  IPS    — inverse-propensity weighted observed outcomes (Stripe's method)
and report the cost of exploring (fraud $ deliberately let through).
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
from src.feedback.exploration import allow_probability, ips_policy_metrics, naive_policy_metrics


def main(args):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    fcfg = cfg["feedback_v2"]
    mode = args.agg_mode or cfg["fe_v2"]["agg_mode"]
    tag = dataset_tag(mode, args.schema or cfg["fe_v2"].get("schema", "full"))
    out = Path(cfg["paths"]["artifacts_v2"].format(mode=tag))
    vp = pd.read_parquet(out / "val_predictions.parquet")
    p, y, amt = vp["p_blend"].to_numpy(), vp["isFraud"].to_numpy().astype(int), vp["TransactionAmt"].to_numpy()

    rng = np.random.default_rng(0)
    current_block = p >= np.quantile(p, 1 - fcfg["sim_current_block_rate"])
    prop = allow_probability(p, current_block, fcfg["explore_q_max"], fcfg["explore_q_min"])
    observed = rng.random(len(p)) < prop
    explored = current_block & observed

    rows = []
    for rate in fcfg["sim_candidate_block_rates"]:
        cand = p >= np.quantile(p, 1 - rate)
        truth_caught = (y * cand).sum()
        truth = {"recall": truth_caught / y.sum(), "precision": truth_caught / cand.sum(), "block_rate": cand.mean()}
        rows.append({"candidate_block_rate": rate,
                     **{f"truth_{k}": round(float(v), 4) for k, v in truth.items()},
                     **{f"naive_{k}": round(v, 4) for k, v in naive_policy_metrics(y, observed, cand).items()},
                     **{f"ips_{k}": round(v, 4) for k, v in ips_policy_metrics(y, observed, prop, cand).items()}})
    res = pd.DataFrame(rows)
    rep = {"explored_transactions": int(explored.sum()),
           "explored_share_of_blocked": float(explored.sum() / max(current_block.sum(), 1)),
           "fraud_amount_let_through_by_exploration": float(amt[explored & (y == 1)].sum()),
           "candidates": res.to_dict("records")}
    (out / "exploration_simulation.json").write_text(json.dumps(rep, indent=1))
    print(res.to_string(index=False))
    print(f"\nExploration cost: {rep['explored_transactions']} explored txns, "
          f"fraud $ let through = {rep['fraud_amount_let_through_by_exploration']:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg-mode", choices=["causal", "transductive"], default=None)
    ap.add_argument("--schema", choices=["full", "raw"], default=None)
    main(ap.parse_args())
