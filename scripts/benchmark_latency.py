"""
Phase 7 — end-to-end latency benchmark of the v2 decision service.

  python scripts/benchmark_latency.py --n 2000                 # real trained bundle (data/processed + artifacts/v2)
  python scripts/benchmark_latency.py --synthetic --n 500      # no trained bundle needed: synthetic IEEE-like world,
                                                               #  real feature engine + real LightGBM + real blender
  python scripts/benchmark_latency.py --synthetic --slow       # force the legacy pandas row path (before-numbers)
  python scripts/benchmark_latency.py --synthetic --profile    # cProfile the scoring loop, print top hotspots

Scores N transactions one by one (features -> model -> rules/policy [+ reasons]) and reports
p50 / p95 / p99 / mean per stage. Target: p50 < 50 ms, p99 < 150 ms single request, warm.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.serving.decision import CostPolicy, DecisionLog, DecisionService, ModelBundle
from src.serving.online_features import OnlineFeatureEngine

STAGES = ["features", "model", "rules_policy", "total"]       # total also covers SHAP reasons (only when action != approve / p high)


def synthetic_world(n_models_trees: int = 300, seed: int = 11):
    """Synthetic IEEE-like world + a real LightGBM bundle on disk. -> (service, stream of raw txn dicts)."""
    import lightgbm as lgb
    from test_serving_online import COSTS, POLICY
    from test_v2_pipeline import _synthetic_ieee
    from src.feature_engineering.pipeline_v2 import run_feature_pipeline_v2
    from src.training.blending import CalibratedRankBlender

    full = _synthetic_ieee(3000, seed)
    tr, va = full[full.TransactionDT < 86400 * 60], full[full.TransactionDT >= 86400 * 60]
    te = _synthetic_ieee(600, seed + 1, start_id=50_000, labelled=False)
    te["TransactionDT"] += 86400 * 90
    data, meta = run_feature_pipeline_v2(tr, va, te, {"label_policy": POLICY})
    feats = meta["feature_cols"]
    X = data["train"][feats].astype(np.float32)
    y = data["train"]["isFraud"].fillna(0).to_numpy()
    m = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 63, "seed": 1, "min_data_in_leaf": 5,
                   "learning_rate": 0.1}, lgb.Dataset(X, y), n_models_trees)
    d = Path(tempfile.mkdtemp(prefix="bench_bundle_"))
    m.save_model(str(d / "lgbm.txt"))
    oof = m.predict(X)
    CalibratedRankBlender({"lgbm": 1.0}).fit({"lgbm": oof}, y).save(d / "blender.pkl")
    (d / "bundle_refit.json").write_text(json.dumps({
        "tag": "synthetic", "features": feats, "cat_cols": [], "model_files": {"lgbm": ["lgbm.txt"]},
        "blender": "blender.pkl", "feature_dtype": "float32"}))
    bundle = ModelBundle(d, "refit")
    eng = OnlineFeatureEngine(meta)
    hist = pd.concat([tr, va], ignore_index=True)
    eng.replay(hist, labels=True, collect=False)
    svc = DecisionService(eng, bundle, CostPolicy(COSTS), log=DecisionLog(None))
    stream = [{k: v for k, v in r.items() if k != "isFraud"} for r in te.to_dict("records")]
    return svc, stream


def run(svc, stream, n):
    lat = {k: [] for k in STAGES}
    for rec in stream[:n]:
        out = svc.decide(rec)
        for k in lat:
            lat[k].append(out["latency_ms"].get(k, 0.0))
    return lat


def report(lat):
    return {k: {**{q: round(float(np.percentile(v, int(q[1:]))), 2) for q in ["p50", "p95", "p99"]},
                "mean": round(float(np.mean(v)), 2)} for k, v in lat.items()}


def main(args):
    if args.slow:
        os.environ["FRAUD_SLOW_ROW_PATH"] = "1"
    if args.synthetic:
        svc, stream = synthetic_world()
        if args.slow:
            svc.engine.fast_rows = False
            svc.bundle.fast_model = False
        warm = [dict(r, TransactionID=r["TransactionID"] + 10_000_000) for r in stream[:5]]
        for r in warm:                                      # warm imports / caches (not reported)
            svc.decide(r)
        stream = (stream * (args.n // len(stream) + 1))[:args.n] if args.n > len(stream) else stream
        if args.n > 600:                                    # TransactionIDs must be unique
            stream = [dict(r, TransactionID=r["TransactionID"] + 1_000_000 * (i // 600), TransactionDT=r["TransactionDT"] + 86400 * 5 * (i // 600))
                      for i, r in enumerate(stream)]
    else:
        from src.serving.service_factory import build_service
        svc = build_service(schema=args.schema, allow_full_schema=args.allow_full_schema)
        svc.log = DecisionLog(None)                       # don't pollute the decision log
        if args.slow:
            svc.engine.fast_rows = False
            svc.bundle.fast_model = False
        val = pd.read_parquet("data/processed/val_raw.parquet").head(args.n)
        stream = val.drop(columns=["isFraud"]).to_dict("records")
    if args.profile:
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.enable()
        lat = run(svc, stream, args.n)
        pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(35)
        pstats.Stats(pr).sort_stats("tottime").print_stats(25)
    else:
        lat = run(svc, stream, args.n)
    rep = report(lat)
    print(json.dumps(rep, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"n": len(lat["total"]), **rep}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--schema", choices=["full", "raw"], default=None, help="default: decision_v2.serving_schema (raw)")
    ap.add_argument("--allow-full-schema", action="store_true", help="replay benchmark on full IEEE columns (not servable live)")
    ap.add_argument("--synthetic", action="store_true", help="synthetic world + real LightGBM (no trained bundle needed)")
    ap.add_argument("--slow", action="store_true", help="legacy pandas row path (for before/after comparison)")
    ap.add_argument("--profile", action="store_true", help="cProfile the scoring loop")
    ap.add_argument("--out", default=None, help="write the latency report as JSON")
    main(ap.parse_args())
