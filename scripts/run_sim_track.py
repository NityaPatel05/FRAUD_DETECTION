"""
Track C (v2) - train & evaluate on the SIMULATED payments world like a real fraud system's offline pipeline.

  python scripts/run_sim_track.py [--data data/synthetic/v2_dense] [--device gpu] [--quick]
                                  [--max-customers N] [--neg-sample 0.25] [--out artifacts/v2/sim]

1. causal features (src/sim_track/features.py): velocity / profile / deviation / sharing / ring-cluster / merchant
   health / account events / labels-as-delivered (reported_label, label_date <= t) / card-testing burst block
2. targets AS KNOWN AT THE CUTOFF (src/sim_track/targets.py): reported_label with label_date <= cutoff, mature
   no-dispute rows = 0, pending = 0 (policy semantics), unobserved (declined) rows excluded, label-delay gap
3. LightGBM (+XGBoost) rolling CV (train | gap | val) -> calibrated rank blend (same code as the IEEE track)
4. deploy at val start -> VAL metrics; deploy at test start (train+val) -> TEST metrics on BOTH truths
   (is_fraud_true = real, reported_label = what the business sees), per-scenario recall, novel scenario
   (bust_out), new customers, CP / CNP / P2P slices
5. dedicated card-testing model on burst features vs the general model; routed combination
Writes <out>/{metrics.json, importance.csv, test_predictions.parquet}. Simulated data: an engineering
demonstration, never real-world accuracy.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import average_precision_score

from src.sim_track.evaluate import pct_vs_reference, recall_for, sim_report
from src.sim_track.features import BURST_PREFIX, build_sim_features
from src.sim_track.targets import known_target, make_sim_folds
from src.training.blending import CalibratedRankBlender, calibration_report
from src.training.evaluation import search_blend_weights
from src.training.gbdt import feature_importance, fit_model, predict

SPEC_CORE = ["amount_usd", "local_hour", "merchant_age_days", "account_age_days", "card_id_prior_declines_1h",
             "card_id_n_1h", "customer_id_new_merchant_id", "ip_n_1h", "device_id_n_1h", "merchant_id_fraud_rate",
             "card_bin_fraud_rate", "acct_sec_since_login_ok"]


def sim_params(tcfg, name, quick):
    """CPU-friendly defaults for the sim world (the IEEE-sized configs are for 590k x 400 and a GPU)."""
    p = dict(tcfg[name])
    if name == "lgbm":
        p.update(learning_rate=0.08 if quick else 0.05, num_leaves=127, min_child_samples=50, colsample_bytree=0.5,
                 n_estimators=700)
    elif name == "xgb":
        p.update(learning_rate=0.08 if quick else 0.05, max_depth=8, n_estimators=700)
    return p


def neg_sample(idx, y, frac, seed):
    if frac >= 1:
        return idx, y
    rng = np.random.default_rng(seed)
    keep = (y == 1) | (rng.random(len(y)) < frac)
    return idx[keep], y[keep]


def run(args):
    t_start = time.time()
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    tcfg = cfg["training_v2"]
    data_dir = Path(args.data)
    meta = json.loads((data_dir / "metadata.json").read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(data_dir / "transactions.parquet")
    ev = pd.read_parquet(data_dir / "events.parquet", columns=["event_time", "customer_id", "event_type", "success",
                                                              "device_id", "ip"])
    if args.max_customers:
        rng = np.random.default_rng(0)
        cu = raw["customer_id"].dropna().unique()
        keep = set(rng.choice(cu, min(args.max_customers, len(cu)), replace=False).tolist())
        raw = raw[raw["customer_id"].isin(keep)].reset_index(drop=True)
        ev = ev[ev["customer_id"].isin(keep)]
        logger.warning("SUBSAMPLED to {} customers ({} rows)", len(keep), len(raw))
    df, feats = build_sim_features(raw, ev)
    del raw, ev
    gc.collect()
    burst = [c for c in df.columns if c.startswith(BURST_PREFIX)]
    logger.info("features: {} general + {} burst | {} rows | {:.0f}s", len(feats), len(burst), len(df), time.time() - t_start)
    Xdf = pd.DataFrame(df[feats].to_numpy(dtype=np.float32), columns=feats)

    sp = {k: pd.Timestamp(v[0]) for k, v in meta["splits"].items()}
    sp_end = {k: pd.Timestamp(v[1]) for k, v in meta["splits"].items()}
    et = df["event_time"]
    gap = args.gap_days
    folds = make_sim_folds(df[et < sp["val"]], sp["val"], 2 if args.quick else tcfg["n_folds"],
                           15 if args.quick else tcfg["fold_val_days"], gap, tcfg["es_frac"])
    y_true = df["is_fraud_true"].to_numpy().astype(int)         # oracle: used ONLY to REPORT dev metrics
    # model selection (early stopping, blend weights, calibration) uses what the business eventually observes:
    # the reported label of rows that got an outcome (declined/unobserved rows have none and are left out)
    observed = df["label_date"].notna().to_numpy()
    y_sel = (df["reported_label"].fillna(0).to_numpy() == 1).astype(int)
    names = ["lgbm"] if args.quick else ["lgbm", "xgb"]
    rep = {"data": str(data_dir), "n_features": len(feats), "n_burst_features": len(burst), "gap_days": gap,
           "neg_sample": args.neg_sample, "max_customers": args.max_customers, "device": args.device,
           "label_semantics": ("target = reported_label with label_date<=cutoff; mature/pending unlabelled = 0; "
                               "unobserved (no label_date) rows excluded; rows within gap_days of cutoff dropped"),
           "note": "simulated data - engineering demonstration, not real-world accuracy", "models": {}}

    oof, best_it = {n: {} for n in names}, {n: [] for n in names}
    for n in names:
        params = sim_params(tcfg, n, args.quick)
        for f in folds:
            tri, ytr = neg_sample(f.train, f.y_train, args.neg_sample, 100 + f.k)
            es = f.es[observed[f.es]]
            m, b = fit_model(n, params, Xdf.iloc[tri], ytr, Xdf.iloc[es], y_sel[es],
                             early_stopping_rounds=60, device=args.device)
            best_it[n].append(b)
            oof[n][f.k] = predict(n, m, Xdf.iloc[f.score])
            del m
            gc.collect()
            logger.info("{} fold {} | train {} (pos {}) | best {} | AP(true) {:.4f} | {:.0f}s", n, f.k, len(tri), int(ytr.sum()),
                        b, average_precision_score(y_true[f.score], oof[n][f.k]), time.time() - t_start)
        rep["models"][n] = {"cv_ap_true": [float(average_precision_score(y_true[f.score], oof[n][f.k])) for f in folds],
                            "best_iter": best_it[n]}
    ks = [f.k for f in folds]
    by_k = {f.k: f for f in folds}
    idx_oof = np.concatenate([by_k[k].score for k in ks])
    fold_id = np.concatenate([np.full(len(by_k[k].score), k) for k in ks])
    oof_p = {n: np.concatenate([oof[n][k] for k in ks]) for n in names}
    sel = observed[idx_oof]
    oof_sel = {n: v[sel] for n, v in oof_p.items()}
    w = search_blend_weights(oof_sel, y_sel[idx_oof][sel], fold_id[sel])
    blender = CalibratedRankBlender(w).fit(oof_sel, y_sel[idx_oof][sel])
    rep["model_selection_labels"] = "reported_label on observed rows (never is_fraud_true)"
    rep["blend_weights"] = w

    def deploy(cutoff, eval_mask, label):
        y, usable = known_target(df["reported_label"], df["label_date"], et, cutoff, gap)
        tr = np.where(usable & (et < cutoff).to_numpy())[0]
        tri, ytr = neg_sample(tr, y[tr].astype(int), args.neg_sample, 7)
        preds = {}
        ix = np.where(eval_mask)[0]
        for n in names:
            nr = int(np.mean(best_it[n]) * 1.2)
            m, _ = fit_model(n, sim_params(tcfg, n, args.quick), Xdf.iloc[tri], ytr, n_rounds=max(nr, 50), device=args.device)
            preds[n] = predict(n, m, Xdf.iloc[ix])
            if n == "lgbm" and label == "test":
                feature_importance(n, m, feats).head(50).to_csv(out_dir / "importance.csv")
            del m
            gc.collect()
        logger.info("deploy {} | train rows {} pos {} | {:.0f}s", label, len(tri), int(ytr.sum()), time.time() - t_start)
        return preds

    va_m = ((et >= sp["val"]) & (et < sp_end["val"])).to_numpy()
    te_m = ((et >= sp["test"]) & (et < sp_end["test"])).to_numpy()
    pv_raw = deploy(sp["val"], va_m, "val")
    pt_raw = deploy(sp["test"], te_m, "test")
    pv, pt = blender.score(pv_raw), blender.score(pt_raw)
    dv, dt = df[va_m].reset_index(drop=True), df[te_m].reset_index(drop=True)
    rep["val"] = sim_report(dv, pv, tcfg["cp_k"])
    rep["test"] = sim_report(dt, pt, tcfg["cp_k"])
    rep["test_calibration_true"] = calibration_report(dt["is_fraud_true"].to_numpy().astype(int), blender.predict_proba(pt_raw))
    obs_t = dt["label_date"].notna().to_numpy()
    rep["test_calibration_reported"] = calibration_report(
        (dt["reported_label"].fillna(0).to_numpy()[obs_t] == 1).astype(int),
        blender.predict_proba({n: v[obs_t] for n, v in pt_raw.items()}))
    for n in names:
        rep["models"][n]["test_ap_true"] = float(average_precision_score(dt["is_fraud_true"], pt_raw[n]))
    logger.info("TEST true AP {:.4f} reported AP {} | novel {}", rep["test"]["true"]["auc_pr"],
                rep["test"].get("reported", {}).get("auc_pr"), rep["test"]["novel"])

    # ---- dedicated card-testing model (burst features), trained on small-amount rows ----
    spec_cols = burst + [c for c in SPEC_CORE if c in df.columns]
    dom = df[BURST_PREFIX + "is_small"].to_numpy() == 1
    Xs = pd.DataFrame(df[spec_cols].to_numpy(dtype=np.float32), columns=spec_cols)
    sparams = dict(tcfg["lgbm"])
    sparams.update(learning_rate=0.05, num_leaves=31, min_child_samples=20, n_estimators=300, colsample_bytree=0.7)

    def spec_deploy(cutoff, eval_mask):
        y, usable = known_target(df["reported_label"], df["label_date"], et, cutoff, gap)
        tr = np.where(usable & dom & (et < cutoff).to_numpy())[0]
        m, _ = fit_model("lgbm", sparams, Xs.iloc[tr], y[tr].astype(int), n_rounds=300, device="cpu")
        ix = np.where(eval_mask)[0]
        return np.where(dom[ix], predict("lgbm", m, Xs.iloc[ix]), 0.0), int(y[tr].sum()), len(tr)

    sv, _, _ = spec_deploy(sp["val"], va_m)
    st, npos, ntr = spec_deploy(sp["test"], te_m)
    # routed combination: general percentile score, small-amount rows may be lifted by the specialist
    # (percentiles vs the VALIDATION score distributions: no test peeking)
    gt = pct_vs_reference(pv, pt)
    spt = pct_vs_reference(sv[dom[va_m]], st)
    comb = np.where(dom[te_m], np.maximum(gt, spt * 0.999), gt)
    ct = {"specialist_train_rows": ntr, "specialist_train_pos": npos,
          "general": recall_for(dt, pt, "card_testing"), "specialist": recall_for(dt, st, "card_testing"),
          "routed_max": recall_for(dt, comb, "card_testing")}
    yy = dt["is_fraud_true"].to_numpy().astype(int)
    for key, sc in [("general", pt), ("specialist", st), ("routed_max", comb)]:
        ct[key]["auc_pr_overall_true"] = float(average_precision_score(yy, sc))
    rep["card_testing_model"] = ct
    logger.info("card_testing recall@1%FPR general {:.3f} | specialist {:.3f} | routed {:.3f}",
                ct["general"]["recall_at_fpr_1pct"], ct["specialist"]["recall_at_fpr_1pct"], ct["routed_max"]["recall_at_fpr_1pct"])

    pd.DataFrame({"transaction_id": dt["transaction_id"], "is_fraud_true": dt["is_fraud_true"],
                  "reported_label": dt["reported_label"], "fraud_scenario": dt["fraud_scenario"], "p_fraud": pt,
                  "p_card_testing_spec": st, "p_routed": comb}).to_parquet(out_dir / "test_predictions.parquet", index=False)
    rep["runtime_min"] = round((time.time() - t_start) / 60, 1)
    (out_dir / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    logger.info("done in {} min -> {}", rep["runtime_min"], out_dir / "metrics.json")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic/v2_dense")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-customers", type=int, default=0)
    ap.add_argument("--neg-sample", type=float, default=0.25)
    ap.add_argument("--gap-days", type=float, default=14.0)
    ap.add_argument("--out", default="artifacts/v2/sim")
    run(ap.parse_args())
