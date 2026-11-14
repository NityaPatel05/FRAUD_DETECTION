"""
GPU smoke test — run this FIRST on the GPU machine (seconds, tiny synthetic data, no real training).

  python scripts/gpu_smoke_test.py                 # expects a GPU; exit code 1 on any problem
  python scripts/gpu_smoke_test.py --device cpu    # logic check on a CPU-only machine
  python scripts/gpu_smoke_test.py --allow-lgbm-cpu  # accept LightGBM on CPU (documented below)

Checks, through the project's OWN training code (src/training/gbdt.py, src/experiments/):
  1. nvidia-smi: GPU name, driver, max CUDA version supported by the driver
  2. torch: CUDA build, torch.cuda.is_available(), GRU forward/backward on the device
  3. XGBoost  device=cuda, tree_method=hist  (+ predict without the device-mismatch warning)
  4. CatBoost task_type=GPU with categorical features (+ no CPU-metric fallback)
  5. LightGBM device_type=gpu → cuda → reports which backend the installed wheel supports
Every stdout/stderr line and Python warning is captured; any line that signals a CPU
fallback or device mismatch makes the test FAIL.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import contextlib
import io
import re
import subprocess
import traceback
import warnings

import numpy as np
import pandas as pd

FALLBACK_PATTERNS = re.compile(
    r"mismatched devices|falling back|fallback|not implemented on GPU|will use CPU|No visible GPU|"
    r"setting .?device.? to CPU|"
    r"GPU Tree Learner was not enabled|CUDA Tree Learner was not enabled|no CUDA|cuda.*not available",
    re.IGNORECASE)


def tiny_data(n=6000, d=20, seed=0):
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    X["cat_a"] = r.integers(0, 30, n).astype(float)
    X.loc[r.random(n) < 0.1, "f3"] = np.nan
    y = ((X["f0"] + 0.5 * X["f1"] + (X["cat_a"] % 7 == 0) + r.normal(0, 1, n)) > 2.2).astype(int).to_numpy()
    return X, y


def captured(fn):
    """Run fn, capture stdout/stderr/warnings; return (result, captured_text, error)."""
    buf_o, buf_e = io.StringIO(), io.StringIO()
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(buf_o), contextlib.redirect_stderr(buf_e):
        warnings.simplefilter("always")
        try:
            res, err = fn(), None
        except Exception:
            res, err = None, traceback.format_exc(limit=3)
    text = buf_o.getvalue() + buf_e.getvalue() + "\n".join(str(x.message) for x in w)
    return res, text, err


def check_nvidia():
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
        name = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
        cuda = re.search(r"CUDA Version:\s*([\d.]+)", out)
        return True, f"{name} | driver CUDA {cuda.group(1) if cuda else '?'}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, "nvidia-smi not found — no NVIDIA driver on this machine"


def check_torch(device):
    import torch
    info = f"torch {torch.__version__} | built with CUDA {torch.version.cuda} | cuda available={torch.cuda.is_available()}"
    if device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(info + " → install a CUDA build of torch (see docs/GPU_SETUP.md)")
        info += f" | {torch.cuda.get_device_name(0)}"
    from src.experiments.sequence_embedding import _model
    dev = torch.device("cuda" if device == "gpu" else "cpu")
    net = _model(5).to(dev)
    x, m = torch.randn(64, 9, 5, device=dev), torch.ones(64, 9, device=dev)
    loss = net(x, m).pow(2).mean(); loss.backward()
    assert next(net.parameters()).device.type == dev.type
    return info + f" | GRU fwd/bwd on {dev.type} OK"


def check_model(name, device, cfg_params, X, y):
    from src.training.gbdt import fit_model, predict
    tr, va = slice(0, 4500), slice(4500, None)
    params = dict(cfg_params)
    params.update({"n_estimators": 60} if name in ("lgbm", "xgb") else {"iterations": 60})
    m, best = fit_model(name, params, X.iloc[tr], y[tr], X.iloc[va], y[va], early_stopping_rounds=20,
                        cat_cols=["cat_a"] if name == "catboost" else None, device=device)
    p = predict(name, m, X.iloc[va], ["cat_a"] if name == "catboost" else None)
    assert np.isfinite(p).all() and len(p) == len(X) - 4500
    used = getattr(m, "device_used_", "?")
    if name == "catboost":
        used = m.get_all_params().get("task_type", "?")
    if name == "xgb":
        used = str(m.get_params().get("device") or "cpu")
        if used.startswith("cuda") and not check_nvidia()[0]:
            used = "cpu (no visible NVIDIA GPU — XGBoost silently fell back)"
    return f"{name}: trained ({best} rounds), device used = {used}", used


def main(args):
    import yaml
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))["training_v2"]
    X, y = tiny_data()
    failures, lines = [], []

    ok, msg = check_nvidia()
    lines.append(f"[nvidia] {msg}")
    if args.device == "gpu" and not ok:
        failures.append("no NVIDIA GPU/driver")

    for label, fn in [("torch", lambda: check_torch(args.device))] + \
                     [(n, (lambda n=n: check_model(n, args.device, cfg[n], X, y))) for n in ["xgb", "catboost", "lgbm"]]:
        res, text, err = captured(fn)
        if err:
            lines.append(f"[{label}] ERROR\n{err}"); failures.append(f"{label}: error"); continue
        msg_, used = (res, None) if label == "torch" else res
        lines.append(f"[{label}] {msg_}")
        bad = [l for l in text.splitlines() if FALLBACK_PATTERNS.search(l)]
        if args.device == "gpu":
            on_cpu = used is not None and ("cpu" in str(used).lower())
            if label == "lgbm" and on_cpu and args.allow_lgbm_cpu:
                lines.append("        LightGBM on CPU accepted (--allow-lgbm-cpu): wheel has no GPU/CUDA learner")
                bad = [l for l in bad if "Tree Learner" not in l and "falling back" not in l.lower() and "training on CPU" not in l]
            elif on_cpu:
                failures.append(f"{label}: ran on CPU")
        if bad:
            lines.append("        fallback/device warnings:\n          " + "\n          ".join(bad[:5]))
            failures.append(f"{label}: CPU-fallback warning")

    print("\n".join(lines))
    print("\nRESULT:", "PASS — zero errors, no CPU fallback" if not failures else "FAIL → " + "; ".join(failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--allow-lgbm-cpu", action="store_true")
    main(ap.parse_args())
