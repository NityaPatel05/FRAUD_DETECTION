"""
Phase 7 — end-to-end latency benchmark of the v2 decision service (needs trained bundle).

  python scripts/benchmark_latency.py --n 2000

Scores N val transactions one by one (features → model → rules → policy) and reports
p50 / p95 / p99 per stage. Target from the plan: p99 total < 50 ms.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import pandas as pd

from src.serving.decision import DecisionLog
from src.serving.service_factory import build_service


def main(args):
    svc = build_service(schema=args.schema, allow_full_schema=args.allow_full_schema)
    svc.log = DecisionLog(None)                       # don't pollute the decision log
    val = pd.read_parquet("data/processed/val_raw.parquet").head(args.n)
    lat = {"features": [], "model": [], "rules_policy": [], "total": []}
    for rec in val.drop(columns=["isFraud"]).to_dict("records"):
        out = svc.decide(rec)
        for k in lat:
            lat[k].append(out["latency_ms"][k])
    rep = {k: {q: round(float(np.percentile(v, int(q[1:]))), 2) for q in ["p50", "p95", "p99"]} for k, v in lat.items()}
    print(json.dumps(rep, indent=1))
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"n": len(lat["total"]), **rep}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--schema", choices=["full", "raw"], default=None, help="default: decision_v2.serving_schema (raw)")
    ap.add_argument("--allow-full-schema", action="store_true", help="replay benchmark on full IEEE columns (not servable live)")
    ap.add_argument("--out", default=None, help="write the latency report as JSON")
    main(ap.parse_args())
