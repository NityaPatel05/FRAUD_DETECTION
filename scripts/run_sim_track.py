"""
Track C — train & evaluate on the SIMULATED payments world (engineering demonstration).

  python scripts/run_sim_track.py [--data data/synthetic/v1] [--device gpu] [--quick]

1. real-world causal features (src/sim_track/features.py)
2. LightGBM / XGBoost / CatBoost: time-fold CV inside train → refit on train → VAL metrics;
   refit on train+val → TEST metrics (the simulator's test period is labelled)
3. calibrated rank blend (weights from out-of-fold predictions)
4. evaluation that only a simulator can give (ground truth is known):
     recall per fraud scenario at a 1% false-positive rate,
     the NOVEL scenario (bust_out, never seen in train) and
     ring-detection rate for mule rings (ring_id ground truth)
Writes artifacts/v2/sim/{metrics.json, importance.csv, test_predictions.parquet}.
Report these as an engineering demonstration on simulated data, never as real-world accuracy.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import average_precision_score

from src.sim_track.features import build_sim_features
from src.training.blending import CalibratedRankBlender, calibration_report
from src.training.evaluation import score_report, search_blend_weights, time_folds
from src.training.gbdt import feature_importance, fit_model, predict


def threshold_at_fpr(y, p, fpr=0.01):
    legit = p[y == 0]
    return float(np.quantile(legit, 1 - fpr)) if len(legit) else 1.0


def scenario_report(df: pd.DataFrame, p: np.ndarray, th: float) -> dict:
    flagged = p >= th
    out = {}
    for sc, g in df.groupby("fraud_scenario"):
        if sc == "legit":
            continue
        m = (df["fraud_scenario"] == sc).to_numpy()
        out[sc] = {"n": int(m.sum()), "recall_at_1pct_fpr": float(flagged[m].mean())}
    rings = df["ring_id"].notna().to_numpy()
    if rings.any():
        r = pd.DataFrame({"ring": df.loc[rings, "ring_id"].to_numpy(), "hit": flagged[rings]})
        per_ring = r.groupby("ring")["hit"].max()
        out["_rings"] = {"n_rings": int(len(per_ring)), "rings_detected": float(per_ring.mean()),
                         "ring_transactions_flagged": float(r["hit"].mean())}
    return out


def main(args):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    tcfg = cfg["training_v2"]
    data_dir = Path(args.data)
    meta = json.loads((data_dir / "metadata.json").read_text())
    out_dir = Path("artifacts/v2/sim"); out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(data_dir / "transactions.parquet")
    feats_df, feats = build_sim_features(raw)
    del raw; gc.collect()                                             # free the raw table

    split = {k: [pd.Timestamp(v[0]), pd.Timestamp(v[1])] for k, v in meta["splits"].items()}
    et = feats_df["event_time"]
    parts = {k: feats_df[(et >= a) & (et < b)].reset_index(drop=True) for k, (a, b) in split.items()}
    del feats_df; gc.collect()
    tr, va, te = parts["train"], parts["val"], parts["test"]
    logger.info("Sim splits: train {} val {} test {} | fraud {:.2%}/{:.2%}/{:.2%}", len(tr), len(va), len(te),
                tr.is_fraud.mean(), va.is_fraud.mean(), te.is_fraud.mean())

    day = ((tr["event_time"] - tr["event_time"].min()).dt.total_seconds() // 86400).to_numpy().astype(int)
    folds = time_folds(day, tcfg["n_folds"], tcfg["fold_val_days"], tcfg["gap_days"])
    y_tr, y_va, y_te = (d["is_fraud"].to_numpy().astype(int) for d in (tr, va, te))
    models = ["lgbm", "xgb"] if args.quick else ["lgbm", "xgb", "catboost"]
    oof, p_val, p_test, rep = {}, {}, {}, {"models": {}, "n_features": len(feats),
                                            "note": "simulated data — engineering demonstration, not real-world accuracy"}
    fold_id = np.full(len(tr), -1)
    for i, (_, vi) in enumerate(folds):
        fold_id[vi] = i
    trva = pd.concat([tr, va], ignore_index=True); y_trva = np.r_[y_tr, y_va]

    for name in models:
        params = dict(tcfg[name])
        if args.quick:
            params["learning_rate"] = params["learning_rate"] * 2.5
        o = np.full(len(tr), np.nan); best, aps, sizes = [], [], []
        for ti, vi in folds:
            m, b = fit_model(name, params, tr.iloc[ti][feats], y_tr[ti], tr.iloc[vi][feats], y_tr[vi],
                             early_stopping_rounds=tcfg["early_stopping_rounds"], device=args.device)
            o[vi] = predict(name, m, tr.iloc[vi][feats]); best.append(b); sizes.append(len(ti))
            aps.append(average_precision_score(y_tr[vi], o[vi]))
            del m; gc.collect()
        oof[name] = o
        n_rounds = int(np.mean(best) * min(len(tr) / np.mean(sizes), tcfg["refit_max_round_scale"]))
        m, _ = fit_model(name, params, tr[feats], y_tr, n_rounds=n_rounds, device=args.device)
        p_val[name] = predict(name, m, va[feats])
        if name == "lgbm":
            feature_importance(name, m, feats).head(40).to_csv(out_dir / "importance.csv")
        del m; gc.collect()
        n2 = int(np.mean(best) * min(len(trva) / np.mean(sizes), tcfg["refit_max_round_scale"]))
        m, _ = fit_model(name, params, trva[feats], y_trva, n_rounds=n2, device=args.device)
        p_test[name] = predict(name, m, te[feats])
        del m; gc.collect()
        rep["models"][name] = {"cv_ap": float(np.mean(aps)), "cv_ap_std": float(np.std(aps)),
                               "val_auc_pr": float(average_precision_score(y_va, p_val[name])),
                               "test_auc_pr": float(average_precision_score(y_te, p_test[name]))}
        logger.info("{}: CV {:.4f} | val {:.4f} | test {:.4f}", name, np.mean(aps),
                    rep["models"][name]["val_auc_pr"], rep["models"][name]["test_auc_pr"])

    msk = fold_id >= 0
    w = search_blend_weights({k: v[msk] for k, v in oof.items()}, y_tr[msk], fold_id[msk])
    blender = CalibratedRankBlender(w).fit({k: v[msk] for k, v in oof.items()}, y_tr[msk])
    pv, pt = blender.predict_proba(p_val), blender.predict_proba(p_test)
    rep["blend_weights"] = w
    rep["val"] = score_report(y_va, pv); rep["test"] = score_report(y_te, pt)
    rep["test_calibration"] = calibration_report(y_te, pt)
    th = threshold_at_fpr(y_te, pt, 0.01)
    rep["test_by_scenario_at_1pct_fpr"] = scenario_report(te, pt, th)
    novel = meta.get("novel_scenarios", [])
    rep["novel_scenarios"] = {s: rep["test_by_scenario_at_1pct_fpr"].get(s) for s in novel}
    pd.DataFrame({"transaction_id": te["transaction_id"], "is_fraud": y_te, "fraud_scenario": te["fraud_scenario"],
                  "p_fraud": pt}).to_parquet(out_dir / "test_predictions.parquet", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    logger.info("SIM val AUC-PR {:.4f} | test AUC-PR {:.4f} | novel {}", rep["val"]["auc_pr"], rep["test"]["auc_pr"],
                rep["novel_scenarios"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic/v1")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args())
