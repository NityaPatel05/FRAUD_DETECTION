"""
Phase 5 — Optuna hyper-parameter tuning on the walk-forward time folds.

  python scripts/run_tuning.py --model lgbm --trials 40 [--device gpu] [--schema raw]
  python scripts/run_training.py --use-tuned        # then train with the result

Objective = mean AUC-PR over the time folds inside the train period. The val holdout
is never seen. Trials are pruned after each fold if they fall below the median.
Structural params are tuned at a fixed, faster learning rate; the final training
keeps the config learning rate and re-picks the number of rounds by early stopping.
Writes artifacts/v2/<tag>/tuned_<model>.json.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import average_precision_score

from src.feature_engineering.pipeline_v2 import dataset_tag
from src.training.evaluation import time_folds
from src.training.gbdt import fit_model, predict


# name: (kind, low, high, log)
SPACES = {
    "lgbm": {"num_leaves": ("int", 64, 768, True), "min_child_samples": ("int", 20, 400, True),
             "colsample_bytree": ("float", 0.2, 0.8, False), "subsample": ("float", 0.5, 1.0, False),
             "reg_lambda": ("float", 1e-3, 30, True), "reg_alpha": ("float", 1e-3, 10, True),
             "min_split_gain": ("float", 0.0, 1.0, False)},
    "xgb": {"max_depth": ("int", 6, 14, False), "min_child_weight": ("float", 0.5, 30, True),
            "colsample_bytree": ("float", 0.2, 0.8, False), "subsample": ("float", 0.5, 1.0, False),
            "reg_lambda": ("float", 1e-3, 30, True), "gamma": ("float", 0.0, 2.0, False)},
    "catboost": {"depth": ("int", 6, 10, False), "l2_leaf_reg": ("float", 1, 30, True),
                 "random_strength": ("float", 0.1, 10, True), "bagging_temperature": ("float", 0.0, 2.0, False)},
}


def space(trial: optuna.Trial, model: str) -> dict:
    out = {}
    for name, (kind, lo, hi, log) in SPACES[model].items():
        out[name] = (trial.suggest_int(name, lo, hi, log=log) if kind == "int"
                     else trial.suggest_float(name, lo, hi, log=log))
    return out


def config_point(base: dict, model: str) -> dict:
    """Current config values that lie inside the search space (enqueued as trial 0)."""
    return {k: base[k] for k, (_, lo, hi, _) in SPACES[model].items() if k in base and lo <= base[k] <= hi}


def main(args):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    tcfg = cfg["training_v2"]
    mode = args.agg_mode or cfg["fe_v2"]["agg_mode"]
    tag = dataset_tag(mode, args.schema or cfg["fe_v2"].get("schema", "full"))
    fdir = Path(cfg["paths"]["processed_v2"].format(mode=tag))
    out = Path(cfg["paths"]["artifacts_v2"].format(mode=tag)); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((fdir / "meta.json").read_text())
    feats = meta["feature_cols"]
    drop_file = out / "dropped_features.json"
    if drop_file.exists():
        dropped = set(json.loads(drop_file.read_text())["dropped"])
        feats = [c for c in feats if c not in dropped]
    cat_cols = [c for c in meta["catboost_cat_cols"] if c in feats]
    train = pd.read_parquet(fdir / "train.parquet", columns=["day", "isFraud"] + feats)
    y = train["isFraud"].to_numpy().astype(int)
    folds = time_folds(train["day"].to_numpy(), tcfg["n_folds"], tcfg["fold_val_days"], tcfg["gap_days"])
    device = args.device or tcfg.get("device", "cpu")
    base = dict(tcfg[args.model]); base["learning_rate"] = args.lr

    def objective(trial):
        params = {**base, **space(trial, args.model)}
        aps = []
        for i, (tr, va) in enumerate(folds):
            m, best = fit_model(args.model, params, train.iloc[tr][feats], y[tr], train.iloc[va][feats], y[va],
                                early_stopping_rounds=100, cat_cols=cat_cols, device=device)
            aps.append(average_precision_score(y[va], predict(args.model, m, train.iloc[va][feats], cat_cols)))
            trial.report(float(np.mean(aps)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_aps", aps)
        return float(np.mean(aps))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    # trial 0 = current config, so tuning can only report an improvement over it
    study.enqueue_trial(config_point(base, args.model))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    best = study.best_trial
    baseline = study.trials[0].value
    res = {"model": args.model, "best_cv_ap": best.value, "baseline_cv_ap": baseline,
           "fold_aps": best.user_attrs.get("fold_aps"), "best_params": best.params,
           "tuned_at_learning_rate": args.lr, "n_trials": len(study.trials)}
    (out / f"tuned_{args.model}.json").write_text(json.dumps(res, indent=1))
    logger.info("Best CV AP {:.4f} (first trial {:.4f}) → {}", best.value, baseline or float("nan"),
                out / f"tuned_{args.model}.json")
    if baseline is not None and best.value - baseline < 0.005:
        logger.warning("Gain < 0.005 (keep rule): prefer the config defaults; delete the tuned file.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lgbm", "xgb", "catboost"], default="lgbm")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--agg-mode", choices=["causal", "transductive"], default=None)
    p.add_argument("--schema", choices=["full", "raw"], default=None)
    p.add_argument("--device", choices=["cpu", "gpu"], default=None)
    main(p.parse_args())
