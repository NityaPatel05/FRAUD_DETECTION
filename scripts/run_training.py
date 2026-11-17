"""
Step 4 -- train the v2 tree ensemble (LightGBM + XGBoost; CatBoost optional).

  python scripts/run_training.py                         # evaluate on the val holdout
  python scripts/run_training.py --models lgbm --fast     # quick iteration (higher lr, 1 seed)
  python scripts/run_training.py --models lgbm xgb catboost   # CatBoost is off by default
  python scripts/run_training.py --final                  # refit on train+val, predict Kaggle test
  python scripts/run_training.py --agg-mode transductive  # use the transductive feature set
  python scripts/run_training.py --train-gap-days 14      # label-policy gap override

Protocol:
  1. Rolling time folds INSIDE the train period: train | gap | val, val = early-stopping part
     + scoring part (early stopping never sees the rows that are scored). Training targets
     are the labels KNOWN at each fold's cutoff (src/labels/policy.py), not the final labels.
     The val holdout is never used for early stopping/tuning.
  2. ONE model per seed is then refit on all train rows (target as known at the val start,
     last `train_gap_days` dropped) and scored on val against the TRUE labels.
  3. No class weighting, no VAE score, no leaky GNN embeddings.
  4. Models are rank-averaged with weights chosen on the out-of-fold scoring windows; the blend
     score (uncalibrated) is what ranks; isotonic + monotone tiebreak gives the probability.
  5. Report AUC-PR (+ base rate) and operational metrics (recall@FPR, value detection rate,
     CP@k per day, false-decline rate) overall AND on slices (client seen / cold-start /
     uid unbuildable / copied positives removed).
  6. --final reuses the CV (best rounds + OOF) of the validation run when the fingerprint
     (features, params, folds, label policy) is identical: no second identical CV.

The legacy VAE / BiLSTM / GNN trainers are still in src/models/ but are no longer
part of this pipeline (they scored near random; see the audit).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import average_precision_score

from src.labels.policy import LabelPolicy, training_target, policy_summary
from src.training.evaluation import (
    make_folds, fold_summary, score_report, sliced_report, slice_masks, copied_positive_flags,
    search_blend_weights, client_smoothing, mean_se,
)
from src.training.gbdt import fit_model, predict, save_model, feature_importance
from src.training.blending import CalibratedRankBlender, calibration_report
from src.utils.run_guard import finish


def load_features(cfg, tag):
    d = Path(cfg["paths"]["processed_v2"].format(mode=tag))
    meta = json.loads((d / "meta.json").read_text())
    data = {s: pd.read_parquet(d / f"{s}.parquet") for s in ["train", "val", "test"]}
    return data, meta


def resolve_policy(cfg, meta, args) -> LabelPolicy:
    """Features were built under meta['label_policy']; training must use the same one."""
    pol = LabelPolicy.from_cfg(meta.get("label_policy") or cfg.get("label_policy"))
    cfg_pol = LabelPolicy.from_cfg(cfg.get("label_policy"))
    if pol.to_dict() != cfg_pol.to_dict() and meta.get("label_policy"):
        diff = {k: (v, cfg_pol.to_dict()[k]) for k, v in pol.to_dict().items() if v != cfg_pol.to_dict()[k]}
        logger.warning("label_policy in config differs from the one the features were built with "
                       "(features, config): {} -> using the features' policy; rebuild features to change it.", diff)
    gap = args.train_gap_days if args.train_gap_days is not None else cfg_pol.train_gap_days
    return pol.replace(train_gap_days=float(gap))


def cv_fingerprint(name, params, feats, tcfg, policy, train, args) -> str:
    blob = json.dumps({"model": name, "params": params, "feats": list(feats), "policy": policy.to_dict(),
                       "folds": [tcfg["n_folds"], tcfg["fold_val_days"], tcfg.get("es_frac", 0.5)],
                       "es": tcfg["early_stopping_rounds"], "n_train": int(len(train)),
                       "t_max": int(train["TransactionDT"].max()), "fast": bool(args.fast)},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def main(args):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    tcfg = cfg["training_v2"]
    mode = args.agg_mode or cfg["fe_v2"]["agg_mode"]
    from src.feature_engineering.pipeline_v2 import dataset_tag
    tag = dataset_tag(mode, args.schema or cfg["fe_v2"].get("schema", "full"))
    out_dir = Path(cfg["paths"]["artifacts_v2"].format(mode=tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    data, meta = load_features(cfg, tag)
    policy = resolve_policy(cfg, meta, args)
    logger.info("Label policy: {}", policy_summary(policy))
    feats = list(meta["feature_cols"])
    drop_file = out_dir / "dropped_features.json"
    if drop_file.exists() and not args.no_selection:
        dropped = set(json.loads(drop_file.read_text())["dropped"])
        feats = [c for c in feats if c not in dropped]
        logger.info("Time-consistency selection: dropped {} features -> {} left", len(dropped), len(feats))
    cat_cols = [c for c in meta["catboost_cat_cols"] if c in feats]

    train, val, test = data["train"], data["val"], data["test"]
    gap = policy.train_gap_days
    if args.final:
        fit_all, score_df, score_name = pd.concat([train, val], ignore_index=True), test, "test"
    else:
        fit_all, score_df, score_name = train, val, "val"
    cutoff = int(score_df["TransactionDT"].min())               # the model is deployed here
    y_target, usable = training_target(fit_all["isFraud"].to_numpy(), fit_all["TransactionID"].to_numpy(),
                                       fit_all["TransactionDT"].to_numpy(), cutoff, policy, gap)
    fit_df = fit_all[usable].reset_index(drop=True)
    y_fit = y_target[usable].astype(int)
    logger.info("Refit set: {} of {} rows (dropped the last {}d before cutoff); fraud labels known at cutoff: "
                "{} of {} true frauds", len(fit_df), len(fit_all), gap, int(y_fit.sum()),
                int(np.nansum(fit_all.loc[usable, "isFraud"])))

    models = args.models or tcfg["models"]
    device = args.device or tcfg.get("device", "cpu")
    logger.info("Training device: {} | models: {}", device, models)
    n_seeds = 1 if args.fast else (args.n_seeds or tcfg["n_seeds"])

    y_true_tr = train["isFraud"].to_numpy().astype(int)
    folds = make_folds(train["TransactionDT"].to_numpy(), y_true_tr, train["TransactionID"].to_numpy(),
                       tcfg["n_folds"], tcfg["fold_val_days"], policy, gap, tcfg.get("es_frac", 0.5))
    fold_info = [fold_summary(f, y_true_tr) for f in folds]
    for fi in fold_info:
        logger.info("Fold {}: train {} rows ({} known fraud) | ES {} rows | score {} rows (days {}-{}, base rate {:.4f})",
                    fi["fold"], fi["n_train"], fi["train_pos_known"], fi["n_es"], fi["n_score"],
                    fi["val_days"][0], fi["val_days"][1], fi["score_base_rate"])

    X_tr_all = train[feats]
    fold_id = np.full(len(train), -1)
    for f in folds:
        fold_id[f.score] = f.k

    report = {"agg_mode": mode, "n_features": len(feats), "label_policy": policy.to_dict(),
              "folds": fold_info, "models": {}}
    oof, scored = {}, {}
    model_files = {}
    for name in models:
        params = dict(tcfg[name])
        tuned = out_dir / f"tuned_{name}.json"
        if args.use_tuned and tuned.exists():                    # Phase 5: Optuna result
            params.update(json.loads(tuned.read_text())["best_params"])
            logger.info("{}: using tuned params from {}", name, tuned)
        if args.fast:
            params["learning_rate"] = params["learning_rate"] * 2.5
        t0 = time.time()
        fp = cv_fingerprint(name, params, feats, tcfg, policy, train, args)
        cache_json, cache_oof = out_dir / f"cv_cache_{name}.json", out_dir / f"cv_oof_{name}.npy"

        cached = None
        if args.final and not args.no_reuse_cv and cache_json.exists() and cache_oof.exists():
            c = json.loads(cache_json.read_text())
            if c.get("fingerprint") == fp:
                cached = c
        if cached:
            logger.info("{}: REUSING cross-validation of the validation run (fingerprint {}): rounds {}",
                        name, fp, cached["best_rounds"])
            oof_p = np.load(cache_oof)
            best_rounds, fold_aps = cached["best_rounds"], cached["fold_aps"]
            fold_train_sizes, fold_ap_base = cached["fold_train_sizes"], cached["fold_ap_over_base"]
        else:
            logger.info("=" * 60 + f"\n{name}: rolling time-fold CV\n" + "=" * 60)
            oof_p = np.full(len(train), np.nan)
            best_rounds, fold_aps, fold_train_sizes, fold_ap_base = [], [], [], []
            for f in folds:
                m, best = fit_model(name, params, X_tr_all.iloc[f.train], f.y_train,
                                    X_tr_all.iloc[f.es], y_true_tr[f.es],       # ES rows are NOT scored
                                    early_stopping_rounds=tcfg["early_stopping_rounds"],
                                    seed=42, cat_cols=cat_cols, device=device)
                oof_p[f.score] = predict(name, m, X_tr_all.iloc[f.score], cat_cols)
                ap = average_precision_score(y_true_tr[f.score], oof_p[f.score])
                base = float(y_true_tr[f.score].mean())
                best_rounds.append(best); fold_aps.append(float(ap)); fold_train_sizes.append(len(f.train))
                fold_ap_base.append(float(ap / base) if base > 0 else float("nan"))
                logger.info("{} fold {}: AP {:.4f} (base rate {:.4f}, AP/base {:.1f}) best rounds {}",
                            name, f.k, ap, base, fold_ap_base[-1], best)
            cache_json.write_text(json.dumps({"fingerprint": fp, "best_rounds": best_rounds, "fold_aps": fold_aps,
                                              "fold_train_sizes": fold_train_sizes,
                                              "fold_ap_over_base": fold_ap_base}))
            np.save(cache_oof, oof_p)
        oof[name] = oof_p

        # rounds for the refit, scaled with the larger training set (capped)
        scale = min(len(fit_df) / np.mean(fold_train_sizes), tcfg["refit_max_round_scale"])
        n_rounds = int(np.mean(best_rounds) * scale)
        ap_mean, ap_se = mean_se(fold_aps)
        logger.info("{}: CV AP {:.4f} +- {:.4f} (SE) | refit with {} rounds x {} seeds on {} rows",
                    name, ap_mean, ap_se, n_rounds, n_seeds, len(fit_df))

        # -- refit on all fit rows, seed-bagged --------------------------------
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
            "cv_ap_mean": ap_mean, "cv_ap_se": ap_se, "cv_ap_std": float(np.std(fold_aps)),
            "cv_ap_folds": fold_aps, "cv_ap_over_base_folds": fold_ap_base, "best_rounds": best_rounds,
            "refit_rounds": n_rounds, "n_seeds": n_seeds, "cv_reused": bool(cached),
            "minutes": round((time.time() - t0) / 60, 1),
        }

    # -- blend weights chosen on the OOF scoring windows + calibrated percentile blender --
    m = fold_id >= 0
    weights = search_blend_weights({k: v[m] for k, v in oof.items()}, y_true_tr[m], fold_id[m])
    report["blend_weights"] = weights
    logger.info("Blend weights (from OOF): {} (models with weight 0 are dropped)", weights)
    blender = CalibratedRankBlender(weights).fit({k: v[m] for k, v in oof.items()}, y_true_tr[m])
    s_blend = blender.score(scored)                        # uncalibrated: THE ranking score
    p_blend = blender.calibrate(s_blend)                   # calibrated probability (order-identical)
    blender.save(out_dir / f"blender_{'final' if args.final else 'refit'}.pkl")

    # per-fold OOF blend AP (paired, same windows as the model CV numbers)
    fold_blend_ap = []
    for f in folds:
        sc = blender.score({k: v[f.score] for k, v in oof.items()})
        fold_blend_ap.append(float(average_precision_score(y_true_tr[f.score], sc)))
    bm, bse = mean_se(fold_blend_ap)
    report["blend_cv_ap_mean"], report["blend_cv_ap_se"], report["blend_cv_ap_folds"] = bm, bse, fold_blend_ap

    # -- client smoothing post-process (causal running mean per client) -------
    smoothed = client_smoothing(s_blend, score_df["uid1_code"].to_numpy(),
                                score_df["TransactionDT"].to_numpy(), mode=mode)

    bundle = {"tag": tag, "agg_mode": mode, "features": feats, "cat_cols": cat_cols,
              "model_files": {k: v for k, v in model_files.items() if blender.weights.get(k, 0) > 0},
              "blender": f"blender_{'final' if args.final else 'refit'}.pkl",
              "label_policy": policy.to_dict(),
              "use_client_smoothing": False, "trained_on": score_name != "val" and "train+val" or "train"}

    if args.final:
        pd.DataFrame({"TransactionID": score_df["TransactionID"], "isFraud": p_blend}).to_csv(
            out_dir / "submission_blend.csv", index=False)
        pd.DataFrame({"TransactionID": score_df["TransactionID"], "isFraud": smoothed}).to_csv(
            out_dir / "submission_blend_smoothed.csv", index=False)
        logger.info("Test predictions written to {}", out_dir)
    else:
        y_val = score_df["isFraud"].to_numpy().astype(int)
        client = score_df["uid1_code"].to_numpy()
        amt = score_df["TransactionAmt"].to_numpy()
        day = (score_df["TransactionDT"].to_numpy() // 86400)
        k_cp = int(tcfg.get("cp_k", 100))
        for name in scored:
            report["models"][name]["val"] = score_report(y_val, scored[name], client, amt, day, k_cp)
            logger.info("{} VAL AUC-PR {:.4f}", name, report["models"][name]["val"]["auc_pr"])
        report["blend_val"] = score_report(y_val, s_blend, client, amt, day, k_cp)       # ranking score
        report["blend_val_prob"] = score_report(y_val, p_blend, client, amt, day, k_cp)   # calibrated p: same order
        report["blend_val_calibration"] = calibration_report(y_val, p_blend)
        # slices: seen vs cold-start, unbuildable uid, copied positives removed
        both = pd.concat([train, val], ignore_index=True)
        copied = copied_positive_flags(both["uid1_code"].to_numpy(), both["TransactionDT"].to_numpy(),
                                       both["TransactionID"].to_numpy(), both["isFraud"].fillna(0).to_numpy())[len(train):]
        masks = slice_masks(client, fit_df["uid1_code"].to_numpy(), copied)
        report["blend_val_slices"] = sliced_report(y_val, s_blend, masks, amt, client, day, k_cp)
        report["copied_positive_share_val"] = float(copied.sum() / max(y_val.sum(), 1))
        report["slice_sizes"] = {k: int(v.sum()) for k, v in masks.items()}
        if tcfg.get("postprocess_uid_smoothing", True):
            report["blend_val_client_smoothed"] = score_report(y_val, smoothed, client, amt, day, k_cp)
            bundle["use_client_smoothing"] = bool(report["blend_val_client_smoothed"]["auc_pr"] > report["blend_val"]["auc_pr"])
        vp = pd.DataFrame({"TransactionID": score_df["TransactionID"], "TransactionDT": score_df["TransactionDT"],
                           "TransactionAmt": score_df["TransactionAmt"], "uid1_code": score_df["uid1_code"],
                           "isFraud": y_val, "p_blend": p_blend, "s_blend": s_blend, "p_smoothed": smoothed,
                           "copied_positive": copied, "cold_start": masks["cold_start"],
                           **{f"p_{k}": v for k, v in scored.items()}})
        vp.to_parquet(out_dir / "val_predictions.parquet", index=False)
        sl = report["blend_val_slices"]
        logger.info("BLEND VAL AUC-PR {:.4f} (base {:.4f}) | cold-start {} | copied-removed {} | Brier {:.4f} ECE {:.4f}",
                    report["blend_val"]["auc_pr"], report["blend_val"]["base_rate"],
                    sl["cold_start"].get("auc_pr"), sl["copied_removed"].get("auc_pr"),
                    report["blend_val_calibration"]["brier"], report["blend_val_calibration"]["ece"])
    (out_dir / f"bundle_{'final' if args.final else 'refit'}.json").write_text(json.dumps(bundle, indent=1))

    report["features"] = feats
    fname = "metrics_final.json" if args.final else "metrics_val.json"
    (out_dir / fname).write_text(json.dumps(report, indent=1, default=float))
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
    p.add_argument("--train-gap-days", type=float, default=None,
                   help="rows this close before the cutoff are not trained on (overrides label_policy.train_gap_days)")
    p.add_argument("--no-reuse-cv", action="store_true", help="--final: redo the CV even if an identical one exists")
    main(p.parse_args())
    finish()
