"""
Step 4 — train the v2 tree ensemble (LightGBM + XGBoost + CatBoost).

  python scripts/run_training.py                         # evaluate on the val holdout
  python scripts/run_training.py --models lgbm --fast     # quick iteration (higher lr, 1 seed)
  python scripts/run_training.py --final                  # refit on train+val, predict Kaggle test
  python scripts/run_training.py --agg-mode transductive  # use the transductive feature set

Protocol (fixes from the audit):
  1. Walk-forward time folds INSIDE the train period (with a gap) choose the number
     of boosting rounds. The val holdout is never used for early stopping/tuning.
  2. ONE model per seed is then refit on ALL train rows (the old code averaged 5
     fold models, most of which never saw the most recent weeks) and scored on val.
  3. No class weighting, no VAE score, no leaky GNN embeddings.
  4. Models are rank-averaged with weights chosen on the out-of-fold predictions
     (no stacker trained on fake-zero OOF rows).
  5. Report AUC-PR plus operational metrics; also AUC-PR after client smoothing.

The legacy VAE / BiLSTM / GNN trainers are still in src/models/ but are no longer
part of this pipeline (they scored near random; see the audit).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import average_precision_score

from src.training.evaluation import (
    time_folds, score_report, search_blend_weights, client_smoothing,
)
from src.training.gbdt import fit_model, predict, save_model, feature_importance
from src.training.blending import CalibratedRankBlender, calibration_report


def load_features(cfg, tag):
    d = Path(cfg["paths"]["processed_v2"].format(mode=tag))
    meta = json.loads((d / "meta.json").read_text())
    data = {s: pd.read_parquet(d / f"{s}.parquet") for s in ["train", "val", "test"]}
    return data, meta


def main(args):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    tcfg = cfg["training_v2"]
    mode = args.agg_mode or cfg["fe_v2"]["agg_mode"]
    from src.feature_engineering.pipeline_v2 import dataset_tag
    tag = dataset_tag(mode, args.schema or cfg["fe_v2"].get("schema", "full"))
    out_dir = Path(cfg["paths"]["artifacts_v2"].format(mode=tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    data, meta = load_features(cfg, tag)
    feats = list(meta["feature_cols"])
    drop_file = out_dir / "dropped_features.json"
    if drop_file.exists() and not args.no_selection:
        dropped = set(json.loads(drop_file.read_text())["dropped"])
        feats = [c for c in feats if c not in dropped]
        logger.info("Time-consistency selection: dropped {} features → {} left", len(dropped), len(feats))
    cat_cols = [c for c in meta["catboost_cat_cols"] if c in feats]

    train, val, test = data["train"], data["val"], data["test"]
    if args.final:
        fit_df, score_df, score_name = pd.concat([train, val], ignore_index=True), test, "test"
    else:
        fit_df, score_df, score_name = train, val, "val"
    y_fit = fit_df["isFraud"].to_numpy().astype(int)

    models = args.models or tcfg["models"]
    device = args.device or tcfg.get("device", "cpu")
    logger.info("Training device: {}", device)
    n_seeds = 1 if args.fast else (args.n_seeds or tcfg["n_seeds"])
    folds = time_folds(train["day"].to_numpy(), tcfg["n_folds"], tcfg["fold_val_days"], tcfg["gap_days"])
    for i, (tr, va) in enumerate(folds):
        logger.info("Fold {}: train {} rows (day ≤ {}), val {} rows (days {}–{})", i, len(tr),
                    train["day"].iloc[tr].max(), len(va), train["day"].iloc[va].min(), train["day"].iloc[va].max())

    X_tr_all = train[feats]
    y_tr_all = train["isFraud"].to_numpy().astype(int)
    fold_id = np.full(len(train), -1)
    for i, (_, va) in enumerate(folds):
        fold_id[va] = i

    report = {"agg_mode": mode, "n_features": len(feats), "folds": [], "models": {}}
    oof, scored = {}, {}

    model_files = {}
    for name in models:
        params = dict(tcfg[name])
        tuned = out_dir / f"tuned_{name}.json"
        if args.use_tuned and tuned.exists():                    # Phase 5: Optuna result
            params.update(json.loads(tuned.read_text())["best_params"])
            logger.info("{}: using tuned params from {}", name, tuned)
        if args.fast:
            key = "learning_rate"
            params[key] = params[key] * 2.5
        t0 = time.time()
        logger.info("=" * 60 + f"\n{name}: time-fold CV\n" + "=" * 60)

        # ── 1. CV inside train → best rounds + OOF predictions ────────────
        oof_p = np.full(len(train), np.nan)
        best_rounds, fold_aps, fold_train_sizes = [], [], []
        for i, (tr, va) in enumerate(folds):
            m, best = fit_model(name, params, X_tr_all.iloc[tr], y_tr_all[tr],
                                X_tr_all.iloc[va], y_tr_all[va],
                                early_stopping_rounds=tcfg["early_stopping_rounds"],
                                seed=42, cat_cols=cat_cols, device=device)
            oof_p[va] = predict(name, m, X_tr_all.iloc[va], cat_cols)
            ap = average_precision_score(y_tr_all[va], oof_p[va])
            best_rounds.append(best); fold_aps.append(ap); fold_train_sizes.append(len(tr))
            logger.info("{} fold {}: AP {:.4f} (best rounds {})", name, i, ap, best)
        oof[name] = oof_p

        # rounds for the refit, scaled with the larger training set (capped)
        scale = min(len(fit_df) / np.mean(fold_train_sizes), tcfg["refit_max_round_scale"])
        n_rounds = int(np.mean(best_rounds) * scale)
        logger.info("{}: CV AP {:.4f} ± {:.4f} | refit with {} rounds × {} seeds on {} rows",
                    name, np.mean(fold_aps), np.std(fold_aps), n_rounds, n_seeds, len(fit_df))

        # ── 2. refit on all fit rows, seed-bagged ─────────────────────────
        preds = []
        for k in range(n_seeds):
            m, _ = fit_model(name, params, fit_df[feats], y_fit, n_rounds=n_rounds,
                             seed=42 + k, cat_cols=cat_cols, device=device)
            preds.append(predict(name, m, score_df[feats], cat_cols))
            path = save_model(name, m, out_dir / f"{name}_{'final' if args.final else 'refit'}_seed{k}")
            model_files.setdefault(name, []).append(path.name)
            if k == 0:
                feature_importance(name, m, feats).to_csv(out_dir / f"{name}_importance.csv")
        scored[name] = np.mean(preds, axis=0)

        report["models"][name] = {
            "cv_ap_mean": float(np.mean(fold_aps)), "cv_ap_std": float(np.std(fold_aps)),
            "cv_ap_folds": [float(a) for a in fold_aps], "best_rounds": best_rounds,
            "refit_rounds": n_rounds, "n_seeds": n_seeds, "minutes": round((time.time() - t0) / 60, 1),
        }
        if not args.final:
            report["models"][name]["val"] = score_report(
                score_df["isFraud"].to_numpy(), scored[name], score_df["uid1_code"].to_numpy())
            logger.info("{} VAL AUC-PR {:.4f}", name, report["models"][name]["val"]["auc_pr"])

    # ── 3. blend weights chosen on OOF folds + calibrated percentile blender ─
    m = fold_id >= 0
    weights = search_blend_weights({k: v[m] for k, v in oof.items()}, y_tr_all[m], fold_id[m])
    report["blend_weights"] = weights
    logger.info("Blend weights (from OOF): {} (models with weight 0 are dropped)", weights)
    blender = CalibratedRankBlender(weights).fit({k: v[m] for k, v in oof.items()}, y_tr_all[m])
    p_blend = blender.predict_proba(scored)                     # calibrated fraud probability
    blender.save(out_dir / f"blender_{'final' if args.final else 'refit'}.pkl")

    # ── 4. client smoothing post-process (causal running mean per client) ───
    smoothed = client_smoothing(p_blend, score_df["uid1_code"].to_numpy(),
                                score_df["TransactionDT"].to_numpy(), mode=mode)

    # ── 5. bundle manifest for serving (Phase 7 loads exactly this) ─────────
    bundle = {"tag": tag, "agg_mode": mode, "features": feats, "cat_cols": cat_cols,
              "model_files": {k: v for k, v in model_files.items() if blender.weights.get(k, 0) > 0},
              "blender": f"blender_{'final' if args.final else 'refit'}.pkl",
              "use_client_smoothing": False, "trained_on": score_name != "val" and "train+val" or "train"}

    if args.final:
        pd.DataFrame({"TransactionID": score_df["TransactionID"], "isFraud": p_blend}).to_csv(
            out_dir / "submission_blend.csv", index=False)
        pd.DataFrame({"TransactionID": score_df["TransactionID"], "isFraud": smoothed}).to_csv(
            out_dir / "submission_blend_smoothed.csv", index=False)
        logger.info("Test predictions written to {}", out_dir)
    else:
        y_val = score_df["isFraud"].to_numpy()
        report["blend_val"] = score_report(y_val, p_blend, score_df["uid1_code"].to_numpy())
        report["blend_val_calibration"] = calibration_report(y_val, p_blend)
        if tcfg.get("postprocess_uid_smoothing", True):
            report["blend_val_client_smoothed"] = score_report(y_val, smoothed, score_df["uid1_code"].to_numpy())
            # gate: serve smoothing only if it actually helped on val
            bundle["use_client_smoothing"] = bool(report["blend_val_client_smoothed"]["auc_pr"] > report["blend_val"]["auc_pr"])
        # per-row predictions for the policy / exploration scripts (Phase 7–8)
        vp = pd.DataFrame({"TransactionID": score_df["TransactionID"], "TransactionDT": score_df["TransactionDT"],
                           "TransactionAmt": score_df["TransactionAmt"], "uid1_code": score_df["uid1_code"],
                           "isFraud": y_val, "p_blend": p_blend, "p_smoothed": smoothed,
                           **{f"p_{k}": v for k, v in scored.items()}})
        vp.to_parquet(out_dir / "val_predictions.parquet", index=False)
        logger.info("BLEND VAL AUC-PR {:.4f} | Brier {:.4f} ECE {:.4f} | client-smoothed {:.4f}",
                    report["blend_val"]["auc_pr"], report["blend_val_calibration"]["brier"],
                    report["blend_val_calibration"]["ece"],
                    report.get("blend_val_client_smoothed", {}).get("auc_pr", float("nan")))
    (out_dir / f"bundle_{'final' if args.final else 'refit'}.json").write_text(json.dumps(bundle, indent=1))

    report["features"] = feats
    fname = "metrics_final.json" if args.final else "metrics_val.json"
    (out_dir / fname).write_text(json.dumps(report, indent=1))
    logger.info("Report saved to {}", out_dir / fname)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="v2 tree-ensemble training")
    p.add_argument("--agg-mode", choices=["causal", "transductive"], default=None)
    p.add_argument("--schema", choices=["full", "raw"], default=None)
    p.add_argument("--models", nargs="+", choices=["lgbm", "xgb", "catboost"], default=None)
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--fast", action="store_true", help="2.5x learning rate, 1 seed (for iteration)")
    p.add_argument("--final", action="store_true", help="refit on train+val and predict Kaggle test")
    p.add_argument("--no-selection", action="store_true", help="ignore dropped_features.json")
    p.add_argument("--device", choices=["cpu", "gpu"], default=None, help="overrides training_v2.device")
    p.add_argument("--use-tuned", action="store_true", help="merge artifacts/v2/<tag>/tuned_<model>.json (run_tuning.py)")
    main(p.parse_args())
