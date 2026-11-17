"""
Phase C2: closed-loop experiments on the v2 simulator.

  1. selection bias vs exploration (no exploration / eps 2% / eps 5% with IPW / oracle labels)
  2. retraining under adaptive attackers (no retrain / calendar / drift-triggered)
  3. cost-based decisions (single threshold / per-segment thresholds / expected-cost actions)
  4. novelty layer on the held-out scenario (bust_out)

    venv\\Scripts\\python.exe scripts\\run_closed_loop.py --quick                 # tiny smoke run (~2 min)
    venv\\Scripts\\python.exe scripts\\run_closed_loop.py                         # default: ~35 min CPU, 3 seeds
    venv\\Scripts\\python.exe scripts\\run_closed_loop.py --size full --seeds 1 2 3   # GPU / big machine

Writes artifacts/v2/closed_loop/metrics.json. The learning policy sees only public columns, engine history and the
labels the loop delivered; is_fraud_true is used for scoring outcomes only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.closed_loop.config import COST_DEFAULTS, DELAY_SCALE, DRIFT, MODEL, POLICY   # noqa: E402
from src.closed_loop.experiments import (PERIOD, Setup, exp1_selection_bias, exp2_retraining, exp3_cost,   # noqa: E402
                                         exp4_novelty)

SUMMARY_KEYS = ["fraud_loss_usd", "fraud_block_rate", "false_decline_rate", "false_declines", "false_decline_cost_usd",
                "sca_abandon_cost_usd", "review_cost_usd", "reviews", "stepup_rate", "review_rate",
                "exploration_rows", "exploration_fraud_loss_usd", "exploration_net_cost_usd", "net_cost_usd", "fraud_rows"]


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def aggregate(arms):
    """arm name -> mean over seeds of summaries, snapshot series and aligned series."""
    out = {}
    for name in dict.fromkeys(a["arm"] for a in arms):
        rs = [a for a in arms if a["arm"] == name]
        summ = {k: _mean([r["summary_post_warmup"][k] for r in rs]) for k in SUMMARY_KEYS}
        summ["n_retrains"] = _mean([r["n_retrains"] for r in rs])
        full = {k: _mean([r["summary_all_days"][k] for r in rs]) for k in ("fraud_loss_usd", "net_cost_usd")}
        n_snap = min(len(r["snapshots"]) for r in rs) if rs else 0
        snaps = [dict(day=rs[0]["snapshots"][i]["day"],
                      auc_pr=_mean([r["snapshots"][i]["auc_pr"] for r in rs]),
                      recall_at_1pct_fpr=_mean([r["snapshots"][i]["recall_at_1pct_fpr"] for r in rs]),
                      vdr_at_1pct_fpr=_mean([r["snapshots"][i]["vdr_at_1pct_fpr"] for r in rs])) for i in range(n_snap)]
        days = sorted({x["day0"] for r in rs for x in r["aligned"]})
        aligned = []
        for d0 in days:
            xs = [x for r in rs for x in r["aligned"] if x["day0"] == d0]
            aligned.append(dict(day0=d0, auc_pr=_mean([x["auc_pr"] for x in xs]),
                                recall_at_1pct_fpr=_mean([x["recall_at_1pct_fpr"] for x in xs]), n=len(xs)))
        pdays = sorted({x["day0"] for r in rs for x in r["series"]})
        live = []
        for d0 in pdays:
            xs = [x for r in rs for x in r["series"] if x["day0"] == d0]
            live.append(dict(day0=d0, **{k: _mean([x[k] for x in xs]) for k in
                                         ("fraud_loss_usd", "fraud_block_rate", "false_decline_rate", "exploration_net_cost_usd", "net_cost_usd")}))
        test = {k: _mean([r["summary_test_period"][k] for r in rs]) for k in ("fraud_loss_usd", "fraud_block_rate", "net_cost_usd")}
        def per_seed_last(key, n=1):
            return _mean([_mean([x[key] for x in r["snapshots"][-n:]]) for r in rs if r["snapshots"]])
        warm = min(x["day0"] for r in rs for x in r["aligned"]) if any(r["aligned"] for r in rs) else 0
        al = [_mean([x[k] for x in r["aligned"] if x["day0"] >= warm + 14]) for r in rs for k in ("auc_pr",)]
        al_rec = [_mean([x["recall_at_1pct_fpr"] for x in r["aligned"] if x["day0"] >= warm + 14]) for r in rs]
        out[name] = dict(n_seeds=len(rs), summary=summ, all_days=full, test_period=test, snapshots_mean=snaps,
                         aligned_mean=aligned, live_series_mean=live,
                         # quality of the model each seed ENDED with, and of the deployed model averaged over time
                         final_model=dict(auc_pr=per_seed_last("auc_pr"), recall_at_1pct_fpr=per_seed_last("recall_at_1pct_fpr"),
                                          vdr_at_1pct_fpr=per_seed_last("vdr_at_1pct_fpr")),
                         last3_snapshots_mean=dict(auc_pr=per_seed_last("auc_pr", 3), recall_at_1pct_fpr=per_seed_last("recall_at_1pct_fpr", 3)),
                         deployed_over_time=dict(auc_pr=_mean(al), recall_at_1pct_fpr=_mean(al_rec)),
                         per_seed=[dict(seed=r["seed"], fraud_loss_usd=r["summary_post_warmup"]["fraud_loss_usd"],
                                        net_cost_usd=r["summary_post_warmup"]["net_cost_usd"],
                                        final_auc_pr=r["snapshots"][-1]["auc_pr"] if r["snapshots"] else None,
                                        final_recall=r["snapshots"][-1]["recall_at_1pct_fpr"] if r["snapshots"] else None,
                                        n_retrains=r["n_retrains"]) for r in rs])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--size", default=None, choices=["quick", "default", "full"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--exps", default="1,2,3,4")
    ap.add_argument("--threshold", type=float, default=None, help="decline threshold on p(fraud)")
    ap.add_argument("--out", default="artifacts/v2/closed_loop/metrics.json")
    ap.add_argument("--reaggregate", action="store_true", help="rebuild the summary blocks of an existing metrics.json")
    a = ap.parse_args()
    if a.reaggregate:
        m = json.loads(Path(a.out).read_text())
        for k in ("exp1_selection_bias", "exp2_retraining", "exp3_cost"):
            if k in m:
                m[k]["summary"] = _clean(aggregate(m[k]["per_seed"]))
        Path(a.out).write_text(json.dumps(m, indent=1))
        print("[cl] re-aggregated", a.out)
        return
    size = a.size or ("quick" if a.quick else "default")
    seeds = a.seeds or ([1] if size == "quick" else [1, 2, 3])
    exps = {int(x) for x in a.exps.split(",")}
    t0 = time.time()
    S = Setup(size, seeds, threshold=a.threshold)
    print(f"[cl] size={size} seeds={seeds} exps={sorted(exps)} threshold={S.pol['decline_threshold']}", flush=True)
    S.eval_stream()
    print(f"[cl] eval stream {len(S.eval_df):,} rows, {int(S.eval_df['is_fraud_true'].sum())} fraud  ({time.time() - t0:.0f}s)", flush=True)
    res = dict(meta=dict(size=size, seeds=seeds, world=S.cfg0.to_dict(), policy=S.pol, model=S.model, period_days=PERIOD,
                         delay_scale=DELAY_SCALE, cost_defaults=COST_DEFAULTS, drift=DRIFT,
                         eval_stream=dict(seed=S.eval_seed, rows=int(len(S.eval_df)), fraud=int(S.eval_df["is_fraud_true"].sum()),
                                          fixed_window_days=list(S.fixed_days))))
    kept = []
    e1 = None
    if exps & {1, 2, 4}:
        if 1 in exps or 2 in exps or 4 in exps:
            e1 = exp1_selection_bias(S, keep_tx=True)
            kept += [x for x in e1["arms"] if "_tx" in x]
            res["exp1_selection_bias"] = dict(threshold=e1["threshold"], summary=aggregate(e1["arms"]), per_seed=_clean(e1["arms"]))
            print(f"[cl] exp1 done ({time.time() - t0:.0f}s)", flush=True)
    if 2 in exps or 4 in exps:
        cal14 = [x for x in e1["arms"] if x["arm"] == "explore_2pct"]
        e2 = exp2_retraining(S, reuse_calendar14=[{k: v for k, v in x.items() if not k.startswith("_")} for x in cal14])
        kept += [x for x in e2["arms"] if "_tx" in x]
        res["exp2_retraining"] = dict(threshold=e2["threshold"], explore=e2["explore"], summary=aggregate(e2["arms"]),
                                      per_seed=_clean([x for x in e2["arms"]]))
        print(f"[cl] exp2 done ({time.time() - t0:.0f}s)", flush=True)
    if 3 in exps:
        e3 = exp3_cost(S)
        res["exp3_cost"] = dict(cost=e3["cost"], explore=e3["explore"], summary=aggregate(e3["arms"]), per_seed=_clean(e3["arms"]))
        print(f"[cl] exp3 done ({time.time() - t0:.0f}s)", flush=True)
    if 4 in exps:
        e4 = exp4_novelty(S, kept, S.eval_df)
        res["exp4_novelty"] = _clean(e4)
        print(f"[cl] exp4 done ({time.time() - t0:.0f}s)", flush=True)
    res["meta"]["runtime_sec"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(res), indent=1))
    print(f"[cl] wrote {out}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
