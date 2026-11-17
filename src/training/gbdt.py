"""
One interface for LightGBM / XGBoost / CatBoost in the v2 pipeline.

fit(...) with a validation set  → early stopping on average precision, returns best rounds
fit(...) without one            → fixed number of rounds (refit on all training rows)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _cat_frame(X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """CatBoost needs integer/string categoricals without NaN."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].fillna(-1).astype(np.int64)
    return X


def apply_device(name: str, params: Dict, device: str) -> Dict:
    """Translate device='gpu' into each library's own GPU settings."""
    params = dict(params)
    gpu_max_bin = params.pop("gpu_max_bin", 255)           # our own key, not a LightGBM parameter
    gpu_eval_metric = params.pop("gpu_eval_metric", "Logloss")
    if device != "gpu":
        if name == "lgbm":                     # OpenCL ids mean nothing on CPU
            params.pop("gpu_platform_id", None); params.pop("gpu_device_id", None)
        return params
    if name == "xgb":
        params["device"] = "cuda"; params["tree_method"] = "hist"
    elif name == "catboost":
        params["task_type"] = "GPU"; params.setdefault("devices", "0")
        params.pop("thread_count", None)
        # PRAUC is not available as a GPU eval metric: CatBoost computes it on CPU every
        # iteration (slow + CPU-fallback warning), so early stopping would be CPU-bound.
        # Default = Logloss (GPU-native; a proper scoring rule, correlates with AP but is not
        # AP: rounds may differ slightly from PRAUC). Override with training_v2.catboost.gpu_eval_metric.
        if params.get("eval_metric") in ("PRAUC", "AUC"):
            params["eval_metric"] = gpu_eval_metric
    elif name == "lgbm":
        params["device_type"] = "gpu"          # OpenCL build; fit_model tries cuda, then CPU
        # max_bin 63 (LightGBM's GPU speed tip) quantises features too coarsely for ~400 tabular
        # columns and cost accuracy; default 255 (training_v2.lgbm.gpu_max_bin). Explicit max_bin wins.
        params.setdefault("max_bin", int(gpu_max_bin))
        # OpenCL may list an integrated GPU first: pin the NVIDIA one via config
        # (training_v2.lgbm.gpu_platform_id / gpu_device_id; see docs/GPU_SETUP.md)
    return params


LGBM_GPU_BACKENDS = ("gpu", "cuda")            # OpenCL wheel first, then a CUDA build


def fit_model(
    name: str,
    params: Dict,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: Optional[pd.DataFrame] = None,
    y_va: Optional[np.ndarray] = None,
    early_stopping_rounds: int = 200,
    n_rounds: Optional[int] = None,
    seed: int = 42,
    cat_cols: Optional[List[str]] = None,
    device: str = "cpu",
) -> Tuple[object, int]:
    params = apply_device(name, params, device)
    if name == "lgbm" and device == "gpu":
        from loguru import logger
        errors = []
        for backend in LGBM_GPU_BACKENDS:
            try:
                model, best = _fit(name, {**params, "device_type": backend}, X_tr, y_tr, X_va, y_va,
                                   early_stopping_rounds, n_rounds, seed, cat_cols)
                model.device_used_ = backend
                return model, best
            except Exception as e:             # wheel built without this backend / no device
                errors.append(f"{backend}: {str(e).splitlines()[0]}")
        logger.warning("LightGBM GPU unavailable ({}); training on CPU.", " | ".join(errors))
        params = {k: v for k, v in params.items() if k not in ("device_type", "max_bin")}
        model, best = _fit(name, params, X_tr, y_tr, X_va, y_va, early_stopping_rounds, n_rounds, seed, cat_cols)
        model.device_used_ = "cpu (fallback)"
        return model, best
    model, best = _fit(name, params, X_tr, y_tr, X_va, y_va, early_stopping_rounds, n_rounds, seed, cat_cols)
    try:
        model.device_used_ = "gpu" if device == "gpu" else "cpu"
    except AttributeError:
        pass
    return model, best


def _fit(name, params, X_tr, y_tr, X_va, y_va, early_stopping_rounds, n_rounds, seed, cat_cols):
    params = dict(params)
    has_val = X_va is not None

    if name == "lgbm":
        import lightgbm as lgb
        if n_rounds:
            params["n_estimators"] = n_rounds
        model = lgb.LGBMClassifier(**params, random_state=seed, metric="average_precision")
        if has_val:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                                 lgb.log_evaluation(500)])
            return model, int(model.best_iteration_ or params["n_estimators"])
        model.fit(X_tr, y_tr)
        return model, int(params["n_estimators"])

    if name == "xgb":
        import xgboost as xgb
        if n_rounds:
            params["n_estimators"] = n_rounds
        if has_val:
            model = xgb.XGBClassifier(**params, random_state=seed,
                                      early_stopping_rounds=early_stopping_rounds)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=500)
            return model, int(model.best_iteration) + 1
        model = xgb.XGBClassifier(**params, random_state=seed)
        model.fit(X_tr, y_tr, verbose=False)
        return model, int(params["n_estimators"])

    if name == "catboost":
        from catboost import CatBoostClassifier, Pool
        cat_cols = cat_cols or []
        if n_rounds:
            params["iterations"] = n_rounds
        model = CatBoostClassifier(**params, random_seed=seed)
        tr = Pool(_cat_frame(X_tr, cat_cols), y_tr, cat_features=cat_cols)
        if has_val:
            va = Pool(_cat_frame(X_va, cat_cols), y_va, cat_features=cat_cols)
            model.fit(tr, eval_set=va, early_stopping_rounds=early_stopping_rounds,
                      use_best_model=True)
            return model, int(model.get_best_iteration()) + 1
        model.fit(tr)
        return model, int(params["iterations"])

    raise ValueError(f"unknown model {name}")


def predict(name: str, model, X: pd.DataFrame, cat_cols: Optional[List[str]] = None) -> np.ndarray:
    if name == "catboost":
        X = _cat_frame(X, cat_cols or [])
    if name == "xgb" and str(model.get_params().get("device", "cpu")).startswith("cuda"):
        # Data lives on the CPU (pandas): predicting with a CUDA booster makes XGBoost copy
        # it and warn about mismatched devices. CPU prediction is exact and avoids that.
        model.get_booster().set_param({"device": "cpu"})
    return model.predict_proba(X)[:, 1]


def save_model(name: str, model, path_stem: Path) -> Path:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    if name == "lgbm":
        p = path_stem.with_suffix(".txt"); model.booster_.save_model(str(p))
    elif name == "xgb":
        p = path_stem.with_suffix(".json"); model.save_model(str(p))
    else:
        p = path_stem.with_suffix(".cbm"); model.save_model(str(p))
    return p


def feature_importance(name: str, model, cols: List[str]) -> pd.Series:
    if name == "lgbm":
        imp = model.booster_.feature_importance(importance_type="gain")
    elif name == "xgb":
        imp = model.feature_importances_
    else:
        imp = model.get_feature_importance()
    return pd.Series(imp, index=cols).sort_values(ascending=False)
