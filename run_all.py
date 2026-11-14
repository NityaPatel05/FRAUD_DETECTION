"""
run_all.py — the ONE file to run on the GPU training machine.

On a fresh machine (only Python 3.11-3.13 + an NVIDIA driver needed):
    git clone https://github.com/NityaPatel05/FRAUD_DETECTION.git
    cd FRAUD_DETECTION
    python run_all.py                  # creates venv/, installs CUDA torch + requirements, re-runs itself
                                       # inside the venv, gets the data (Kaggle API if data/raw is empty)

    python run_all.py                  # everything: setup → features → selection → ablation →
                                       #   tuning → training → evaluation → final model → report
    python run_all.py --quick          # same pipeline, much faster (fewer trials/epochs, 1 seed)
    python run_all.py --resume         # continue after an interruption (skips finished steps)
    python run_all.py --only training  # run a single step (names: see --list)
    python run_all.py --list           # show all steps

At the end it writes:
    results/FINAL_REPORT.md   ← all results in one readable page
    results/results.json      ← the same numbers, machine-readable
    results/logs/<step>.log   ← full log of every step

What happens, in order (each step is an existing project script, so nothing is hidden):
  0  setup        install packages if missing (CUDA torch matched to the driver), check data
  1  gpu_check    GPU smoke test → picks device gpu, or falls back to cpu (see --require-gpu)
  2  preprocess   one time split: train ≈ days 1-130, val ≈ 131-182, Kaggle test
  3  features     v2 features, schema=full (all columns) and schema=raw (real-world fields only)
  4  selection    two-window time-consistency feature check (both schemas)
  5  ablation     which feature families earn their place (keep rule, both schemas)
  6  tuning       Optuna on the time folds (LightGBM; + XGBoost/CatBoost unless --quick)
  7  training     LightGBM + XGBoost + CatBoost, calibrated blend — full and raw schema
  8  heldout      does it catch fraud types it never saw? (Monzo method)
  9  novelty      attack detection → mined rules → drift → discovery
 10  policy       cost-based decisions on val: money saved, decline/review rates
 11  exploration  naive vs inverse-propensity evaluation
 12  latency      end-to-end decision latency (p50/p95/p99)
 13  sequence     optional GRU sequence-score experiment (needs CUDA torch; skipped if absent)
 14  final        refit on train+val → Kaggle test predictions (submission CSV)
 15  sim_generate simulated payments world (real-world fields IEEE lacks) → data/synthetic/v1
 16  sim_train    train + evaluate on it: per-scenario recall, novel scenario, ring detection
 17  report       results/FINAL_REPORT.md

THREE DATASET TRACKS, each run end-to-end, each in its own processes:
  full  IEEE-CIS, all columns              (best accuracy benchmark on REAL data)
  raw   IEEE-CIS, real-world fields only   (the model the live system serves)
  sim   simulated world                    (demonstrates IP/device/merchant/ring features;
                                            engineering demo, not real-world accuracy)

MEMORY & CHECKPOINTS
  * Every command runs in its OWN Python process. When it ends, the operating system frees
    all of its RAM and GPU memory — nothing from one step stays in memory for the next.
  * Every command writes its results to disk (parquet / model files / json) and is marked
    done in results/state.json right away. If the machine runs out of memory or the run is
    interrupted, simply run `python run_all.py` again: finished commands are skipped and it
    continues from the first unfinished one (use --fresh to start over).
  * Free RAM / GPU memory is logged before each command; an out-of-memory crash is reported
    as such, with the hint to re-run (resume) or use --quick / --low-memory.
  * --low-memory runs the heavy steps with smaller settings (single seed, fewer tuning trials,
    no transductive variant, fewer latency samples).

Required steps stop the run on failure. Optional steps (heldout, novelty, exploration,
latency, sequence) are logged and the run continues. Nothing in configs/ is modified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
PY = sys.executable
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"
STATE = RESULTS / "state.json"
OLD_BASELINE = {"val_auc_pr_notebook02_days131_182": 0.608, "val_auc_pr_fold_average_days96_130": 0.6145}


# ─────────────────────────────────────────────────────────────────────────────
# bootstrap: create venv/, install everything, re-run this script inside it
# ─────────────────────────────────────────────────────────────────────────────

VENV = ROOT / "venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
TORCH_CUDA_INDEXES = [("12.8", "cu128"), ("12.6", "cu126"), ("12.4", "cu124"), ("12.1", "cu121"), ("11.8", "cu118")]


def _boot_say(msg):
    print(f"[bootstrap] {msg}", flush=True)


def _in_project_venv() -> bool:
    try:
        return Path(sys.prefix).resolve() == VENV.resolve()
    except OSError:
        return False


_DRIVER_TO_CUDA = [(570, "12.8"), (560, "12.6"), (550, "12.4"), (530, "12.1"), (520, "11.8")]


def _parse_cuda_from_nvidia_smi():
    """(cuda_version or None, driver_version or None) — robust to nvidia-smi format changes."""
    import re as _re
    import subprocess as _sp
    try:
        out = _sp.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except (FileNotFoundError, _sp.TimeoutExpired):
        return None, None
    pats = [r"CUDA Version\s*:\s*([\d]+\.[\d]+)", r"CUDA Driver Version\s*:\s*([\d]+\.[\d]+)",
            r"CUDA[^\n:|]*Version\s*:?\s*([\d]+\.[\d]+)"]
    for pat in pats:
        m = _re.search(pat, out)
        if m:
            return m.group(1), None
    try:
        q = _sp.run(["nvidia-smi", "-q"], capture_output=True, text=True, timeout=30).stdout
        for pat in pats:
            m = _re.search(pat, q)
            if m:
                return m.group(1), None
        drv = _sp.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                      capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        drv = drv[0] if drv else None
    except (FileNotFoundError, _sp.TimeoutExpired):
        drv = None
    if drv:
        try:
            major = int(drv.split(".")[0])
            for min_drv, cuda in _DRIVER_TO_CUDA:
                if major >= min_drv:
                    return cuda, drv
        except ValueError:
            pass
    return None, drv


def _driver_cuda_version():
    return _parse_cuda_from_nvidia_smi()[0]


def _req_fingerprint() -> str:
    import hashlib
    h = hashlib.sha256()
    req = ROOT / "requirements-gpu.txt"
    if req.exists():
        h.update(req.read_bytes())
    h.update(str(_driver_cuda_version()).encode())
    return h.hexdigest()[:16]


def bootstrap_venv(argv) -> None:
    """Create ./venv if needed, install CUDA torch + requirements-gpu.txt, then re-exec in the venv."""
    if sys.version_info < (3, 11):
        sys.exit(f"Python {sys.version.split()[0]} is too old: install Python 3.11-3.13 and run again.")
    if not VENV_PY.exists():
        _boot_say(f"creating virtual environment in {VENV} ...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    marker = VENV / ".run_all_installed"
    fp = _req_fingerprint()
    if not marker.exists() or marker.read_text().strip() != fp:
        pip = [str(VENV_PY), "-m", "pip", "install", "--disable-pip-version-check"]
        _boot_say("upgrading pip ...")
        subprocess.run(pip + ["--upgrade", "pip", "wheel", "setuptools"], check=True)
        cuda = _driver_cuda_version()
        if cuda:
            drv = tuple(int(x) for x in cuda.split(".")[:2])
            idx = next((tag for v, tag in TORCH_CUDA_INDEXES if tuple(int(x) for x in v.split(".")) <= drv), None)
            if idx:
                _boot_say(f"NVIDIA driver supports CUDA {cuda} -> installing torch from the {idx} index ...")
                r = subprocess.run(pip + ["torch", "--index-url", f"https://download.pytorch.org/whl/{idx}"])
                if r.returncode != 0:
                    _boot_say("CUDA torch install failed - the optional GRU step will be skipped.")
            else:
                _boot_say(f"driver CUDA {cuda} is older than every current torch wheel - GRU step will be skipped")
        else:
            _boot_say("no NVIDIA driver found - skipping torch (only the optional GRU step needs it)")
        _boot_say("installing requirements-gpu.txt ...")
        subprocess.run(pip + ["-r", str(ROOT / "requirements-gpu.txt")], check=True)
        marker.write_text(fp)
        _boot_say("environment ready.")
    else:
        _boot_say("venv already set up (requirements unchanged).")
    _boot_say(f"re-running inside the venv: {VENV_PY}")
    sys.exit(subprocess.run([str(VENV_PY), str(Path(__file__).resolve()), *argv]).returncode)



# ─────────────────────────────────────────────────────────────────────────────
# small helpers
# ─────────────────────────────────────────────────────────────────────────────

def say(msg: str) -> None:
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(RESULTS / "run_all.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"steps": {}}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=1))


def run_cmd(step: str, cmd: list, log_name: str | None = None) -> int:
    """Run a command, streaming output to the console and to results/logs/<step>.log."""
    log = LOGS / f"{log_name or step}.log"
    say(f"  $ {' '.join(str(c) for c in cmd)}   (log: {log.relative_to(ROOT)})")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== {dt.datetime.now()} :: {' '.join(map(str, cmd))}\n")
        proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env, cwd=ROOT)
        for line in proc.stdout:
            f.write(line)
            if not QUIET_CONSOLE:
                sys.stdout.write("    " + line)
        return proc.wait()


def memory_status() -> str:
    parts = []
    try:
        import psutil
        vm = psutil.virtual_memory()
        parts.append(f"RAM free {vm.available / 1e9:.1f}/{vm.total / 1e9:.1f} GB")
    except ImportError:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
        if out:
            used, total = (float(x) for x in out[0].split(","))
            parts.append(f"GPU mem used {used / 1024:.1f}/{total / 1024:.1f} GB")
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return " | ".join(parts) or "memory info unavailable"


def looks_like_oom(rc: int, log: Path) -> bool:
    if rc in (-9, 137, 3221225477, -1073741819, 3221226505):      # SIGKILL / Windows access-violation / stack
        return True
    try:
        tail = log.read_text(encoding="utf-8", errors="replace")[-6000:]
    except OSError:
        return False
    return any(k in tail for k in ("MemoryError", "Unable to allocate", "out of memory", "CUDA out of memory",
                                   "std::bad_alloc", "Killed"))


def read_json(path) -> dict | None:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except json.JSONDecodeError:
        return None


def art(tag: str) -> Path:
    return ROOT / "artifacts" / "v2" / tag


# ─────────────────────────────────────────────────────────────────────────────
# step 0: setup
# ─────────────────────────────────────────────────────────────────────────────

def driver_cuda() -> tuple[bool, str | None, str]:
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        name = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
        return True, _parse_cuda_from_nvidia_smi()[0], name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, None, "no NVIDIA driver (nvidia-smi not found)"


def step_setup(args) -> dict:
    info = {}
    # packages
    missing = []
    for mod in ["lightgbm", "xgboost", "catboost", "numba", "optuna", "pandas", "pyarrow", "sklearn",
                "yaml", "loguru", "fastapi", "scipy"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing and not args.no_install:
        say(f"Installing missing packages ({', '.join(missing)}) from requirements-gpu.txt …")
        if run_cmd("setup", [PY, "-m", "pip", "install", "-r", "requirements-gpu.txt"]) != 0:
            raise RuntimeError("pip install -r requirements-gpu.txt failed — see results/logs/setup.log")
    elif missing:
        raise RuntimeError(f"Missing packages {missing}; run without --no-install or pip install -r requirements-gpu.txt")

    # CUDA torch (only needed for the optional GRU experiment)
    has_gpu, cuda_ver, gpu_name = driver_cuda()
    info.update({"gpu": gpu_name, "driver_cuda": cuda_ver})
    torch_ok = subprocess.run([PY, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                              capture_output=True).returncode == 0
    if has_gpu and not torch_ok and not args.no_install and not args.skip_torch_install:
        avail = [("12.8", "cu128"), ("12.6", "cu126"), ("12.4", "cu124"), ("12.1", "cu121"), ("11.8", "cu118")]
        try:
            drv = tuple(int(x) for x in (cuda_ver or "0.0").split(".")[:2])
            idx = next(tag for v, tag in avail if tuple(int(x) for x in v.split(".")) <= drv)
            say(f"Installing CUDA build of torch ({idx}) for driver CUDA {cuda_ver} …")
            rc = run_cmd("setup", [PY, "-m", "pip", "install", "--upgrade", "torch",
                                   "--index-url", f"https://download.pytorch.org/whl/{idx}"])
            torch_ok = rc == 0 and subprocess.run(
                [PY, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                capture_output=True).returncode == 0
        except StopIteration:
            say("Driver CUDA too old for current torch wheels — the optional GRU step will be skipped.")
    info["torch_cuda"] = torch_ok

    # data
    raw = ROOT / "data" / "raw"
    needed = ["train_transaction.csv", "train_identity.csv", "test_transaction.csv", "test_identity.csv"]
    if not all((raw / f).exists() for f in needed):
        if shutil.which("dvc"):
            say("Raw CSVs missing — trying `dvc pull data/raw.dvc` …")
            run_cmd("setup", ["dvc", "pull", "data/raw.dvc"])
        if not all((raw / f).exists() for f in needed):
            download_from_kaggle(raw, args)
        if not all((raw / f).exists() for f in needed):
            raise RuntimeError(
                f"Raw data missing in {raw}. Either copy the 4 IEEE-CIS CSVs there ({', '.join(needed)}), "
                "or put your Kaggle API token in ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME/KAGGLE_KEY), "
                "accept the rules at https://www.kaggle.com/c/ieee-fraud-detection, and run again.")
    info["raw_data"] = "ok"
    return info


def download_from_kaggle(raw: Path, args) -> None:
    has_token = (Path.home() / ".kaggle" / "kaggle.json").exists() or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not has_token:
        say("No Kaggle credentials found -> cannot download the data automatically.")
        return
    if not args.no_install:
        run_cmd("setup", [PY, "-m", "pip", "install", "--disable-pip-version-check", "kaggle"])
    raw.mkdir(parents=True, exist_ok=True)
    say("Downloading IEEE-CIS data from Kaggle (~120 MB zip -> 1.3 GB) ...")
    code = ("from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); "
            f"a.competition_download_files('ieee-fraud-detection', path=r'{raw}', quiet=False)")
    if run_cmd("setup", [PY, "-c", code]) != 0:
        say("Kaggle download failed (did you accept the competition rules?) - see results/logs/setup.log")
        return
    import zipfile
    for z in raw.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw)
        z.unlink()
    say("Kaggle data extracted to data/raw/.")


# ─────────────────────────────────────────────────────────────────────────────
# step 1: GPU check → device
# ─────────────────────────────────────────────────────────────────────────────

def step_gpu_check(args) -> dict:
    if args.cpu:
        return {"device": "cpu", "smoke": "skipped (--cpu)"}
    rc = run_cmd("gpu_check", [PY, "scripts/gpu_smoke_test.py", "--allow-lgbm-cpu", "--torch-optional"])
    if rc == 0:
        return {"device": "gpu", "smoke": "PASS"}
    if args.require_gpu:
        raise RuntimeError("GPU smoke test failed and --require-gpu was given — see results/logs/gpu_check.log "
                           "and docs/GPU_SETUP.md")
    say("WARNING: GPU smoke test FAILED → continuing on CPU (results identical, just slower). "
        "See results/logs/gpu_check.log and docs/GPU_SETUP.md.")
    return {"device": "cpu", "smoke": "FAIL (fell back to cpu)"}


# ─────────────────────────────────────────────────────────────────────────────
# pipeline definition
# ─────────────────────────────────────────────────────────────────────────────

def build_steps(args, device: str) -> list:
    dev = ["--device", device]
    quick = args.quick or args.low_memory
    tuning_models = ["lgbm"] if quick else ["lgbm", "xgb", "catboost"]
    trials = "15" if quick else str(args.trials)
    schemas = ["full", "raw"] if not args.full_only else ["full"]
    S = []   # (name, required, [commands])
    S.append(("preprocess", True, [[PY, "scripts/run_preprocessing.py"]]))
    S.append(("features", True, [[PY, "scripts/run_feature_engineering.py", "--schema", s] for s in schemas]
              + ([[PY, "scripts/run_feature_engineering.py", "--agg-mode", "transductive"]] if args.with_transductive else [])))
    S.append(("selection", True, [[PY, "scripts/run_feature_selection.py", "--two-windows", "--schema", s] for s in schemas]))
    if not args.skip_ablation:
        abl_schemas = ["full"] if quick else schemas
        S.append(("ablation", False, [[PY, "scripts/run_ablation.py", "--schema", s, *dev] for s in abl_schemas]))
    S.append(("tuning", False, [[PY, "scripts/run_tuning.py", "--model", m, "--trials", trials, *dev] for m in tuning_models]))
    seeds = ["--n-seeds", "1"] if quick else []
    S.append(("training", True, [[PY, "scripts/run_training.py", "--use-tuned", *seeds, *dev]]
              + [[PY, "scripts/run_training.py", "--schema", s, *seeds, *dev] for s in schemas if s != "full"]
              + ([[PY, "scripts/run_training.py", "--agg-mode", "transductive", *seeds, *dev]] if args.with_transductive else [])))
    S.append(("heldout", False, [[PY, "scripts/run_heldout_pattern_test.py", *dev]]))
    S.append(("novelty", False, [[PY, "scripts/run_novelty_monitoring.py"]]))
    S.append(("policy", False, [[PY, "scripts/run_policy_simulation.py", "--schema", s] for s in schemas]))
    S.append(("exploration", False, [[PY, "scripts/run_exploration_simulation.py"]]))
    if not args.skip_latency:
        S.append(("latency", False, [[PY, "scripts/benchmark_latency.py", "--n", "500" if quick else "2000",
                                      "--out", "results/latency.json"]]))
    if not args.skip_sequence:
        S.append(("sequence", False, [[PY, "scripts/run_sequence_experiment.py", "--epochs", "1" if quick else "3", *dev]]))
    S.append(("final", False, [[PY, "scripts/run_training.py", "--final", "--use-tuned", *seeds, *dev]]))
    if not args.skip_sim:
        sim_gen = [PY, "scripts/run_simulator.py", "--out", "data/synthetic/v1"]
        if quick:
            sim_gen += ["--no-validate"]
        S.append(("sim_generate", False, [sim_gen]))
        S.append(("sim_train", False, [[PY, "scripts/run_sim_track.py", "--data", "data/synthetic/v1", *dev]
                                       + (["--quick"] if quick else [])]))
    return S


# ─────────────────────────────────────────────────────────────────────────────
# report
# ─────────────────────────────────────────────────────────────────────────────

def fmt(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def model_table(m: dict) -> list:
    rows = ["| Model | CV AUC-PR (mean ± std) | Val AUC-PR | Val ROC-AUC | Recall @1% FPR | Precision top 1% |",
            "|---|---|---|---|---|---|"]
    for name, r in (m.get("models") or {}).items():
        v = r.get("val", {})
        rows.append(f"| {name} | {fmt(r.get('cv_ap_mean'))} ± {fmt(r.get('cv_ap_std'))} | {fmt(v.get('auc_pr'))} | "
                    f"{fmt(v.get('auc_roc'))} | {fmt(v.get('recall_at_fpr_1pct'))} | {fmt(v.get('precision_top_1pct'))} |")
    b = m.get("blend_val") or {}
    rows.append(f"| **calibrated blend** | — | **{fmt(b.get('auc_pr'))}** | {fmt(b.get('auc_roc'))} | "
                f"{fmt(b.get('recall_at_fpr_1pct'))} | {fmt(b.get('precision_top_1pct'))} |")
    return rows


def build_report(state: dict) -> None:
    res = {"generated": dt.datetime.now().isoformat(), "environment": state.get("env", {}),
           "old_baseline": OLD_BASELINE, "steps": state["steps"]}
    L = ["# Fraud detection v2 — final results", "",
         f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Device: **{state.get('env', {}).get('device', '?')}** "
         f"({state.get('env', {}).get('gpu', '?')}). Smoke test: {state.get('env', {}).get('smoke', '?')}.", ""]

    full = read_json(art("causal") / "metrics_val.json")
    raw = read_json(art("causal_raw") / "metrics_val.json")
    trans = read_json(art("transductive") / "metrics_val.json")
    res["metrics_full"], res["metrics_raw"], res["metrics_transductive"] = full, raw, trans

    L += ["## Headline (validation holdout = days 131-182, never used for training or tuning)", "",
          "| Model | Val AUC-PR | vs old 0.608 |", "|---|---|---|"]
    for label, m in [("v2 full schema (all IEEE-CIS columns)", full), ("v2 raw schema (real-world fields only — served live)", raw),
                     ("v2 transductive (Kaggle-style, optimistic)", trans)]:
        if m and m.get("blend_val"):
            ap = m["blend_val"]["auc_pr"]
            L.append(f"| {label} | **{ap:.4f}** | {ap - 0.608:+.4f} |")
    L += ["", "Old baseline: 0.608 (notebook 02, same val window, early-stopped on val so optimistic); "
          "0.6145 (old scripted pipeline, days 96-130).", ""]

    for label, m in [("Full schema", full), ("Raw (real-world) schema", raw)]:
        if not m:
            continue
        L += [f"## Models — {label}", ""] + model_table(m)
        cal = m.get("blend_val_calibration") or {}
        sm = m.get("blend_val_client_smoothed") or {}
        L += ["", f"Blend weights (chosen on out-of-fold): `{m.get('blend_weights')}`  ",
              f"Calibration: Brier {fmt(cal.get('brier'))}, ECE {fmt(cal.get('ece'))}  ",
              f"Client-smoothed blend AUC-PR: {fmt(sm.get('auc_pr'))}  ",
              f"Client precision @500: {fmt((m.get('blend_val') or {}).get('client_precision_at_500'))}  ",
              f"Features used: {m.get('n_features')}", ""]

    # selection
    for tag in ["causal", "causal_raw"]:
        d = read_json(art(tag) / "dropped_features.json")
        if d:
            L.append(f"Time-consistency selection ({tag}): dropped {len(d['dropped'])} features: "
                     f"{', '.join(d['dropped'][:15])}{' …' if len(d['dropped']) > 15 else ''}  ")
    L.append("")

    # ablation
    for tag in ["causal", "causal_raw"]:
        files = sorted(glob.glob(str(art(tag) / "ablation_*.csv")))
        if files:
            import pandas as pd
            ab = pd.read_csv(files[-1])
            res[f"ablation_{tag}"] = ab.to_dict("records")
            L += [f"## Feature-family ablation ({tag}) — remove one family, measure the loss", "",
                  "| Family | # cols | CV AUC-PR without it | Gain from family | Folds helped | Verdict |", "|---|---|---|---|---|---|"]
            for _, r in ab.iterrows():
                L.append(f"| {r['family']} | {r['n_removed']} | {r['ap_mean']:.4f} | {r['delta_if_removed']:+.4f} | "
                         f"{r['folds_hurt_by_removal']} | {r['verdict']} |")
            L.append("")

    # tuning
    tun = {m: read_json(art("causal") / f"tuned_{m}.json") for m in ["lgbm", "xgb", "catboost"]}
    tun = {k: v for k, v in tun.items() if v}
    if tun:
        res["tuning"] = tun
        L += ["## Hyper-parameter tuning (time folds)", "", "| Model | Config CV AUC-PR | Tuned CV AUC-PR | Trials |", "|---|---|---|---|"]
        for k, v in tun.items():
            L.append(f"| {k} | {fmt(v.get('baseline_cv_ap'))} | {fmt(v.get('best_cv_ap'))} | {v.get('n_trials')} |")
        L.append("")

    # held-out patterns
    ho = art("causal") / "heldout_ProductCD.csv"
    if ho.exists():
        import pandas as pd
        h = pd.read_csv(ho); res["heldout"] = h.to_dict("records")
        L += ["## Catching fraud it never saw (each ProductCD's fraud removed from training)", "",
              "| Held out | Val frauds | Recall@1%FPR seen | unseen | Retained |", "|---|---|---|---|---|"]
        for _, r in h.iterrows():
            L.append(f"| {r['held_out']} | {r['val_frauds']} | {r['recall@1%FPR_seen']:.3f} | {r['recall@1%FPR_unseen']:.3f} | {r['retained']:.0%} |")
        L.append("")

    # novelty
    nov = ROOT / "artifacts" / "v2" / "novelty"
    if nov.exists():
        import pandas as pd
        n_alerts = len(pd.read_csv(nov / "alerts.csv")) if (nov / "alerts.csv").exists() else None
        drift = read_json(nov / "drift.json") or {}
        rules_txt = (nov / "candidate_rules.yaml").read_text() if (nov / "candidate_rules.yaml").exists() else ""
        res["novelty"] = {"alerts": n_alerts, "candidate_rules": rules_txt.count("- id:"), "drift": drift}
        L += ["## Novelty layer (replaces the VAE)", "",
              f"- Segment attack alerts in the val period: **{n_alerts}**",
              f"- Candidate shadow rules mined & backtested: **{rules_txt.count('- id:')}** (`artifacts/v2/novelty/candidate_rules.yaml`)",
              f"- Drift (train→val) domain-classifier AUC: **{fmt(drift.get('domain_auc'), 3)}**; "
              f"features with PSI > 0.2: {len(drift.get('psi_investigate', []))}", ""]

    # policy
    for tag in ["causal", "causal_raw"]:
        pol = read_json(art(tag) / "policy_report.json")
        if pol:
            res[f"policy_{tag}"] = pol
            L += [f"## Decision policy on val ({tag})", "",
                  f"- Cost with policy: **{pol['cost_policy']:,.0f}** vs approve-everything {pol['cost_approve_all']:,.0f} "
                  f"→ saves **{pol['savings_vs_approve_all']:,.0f}** "
                  f"(plain score threshold with same declines: {pol['cost_score_threshold_same_declines']:,.0f})",
                  f"- Fraud amount intercepted: {pol['fraud_amount_intercepted']:,.0f} of {pol['fraud_amount_total']:,.0f}; "
                  f"fraud recall (any action): {pol['fraud_recall_any_action']:.1%}",
                  f"- Action rates: {pol['action_rates']}; max reviews/day {pol['reviews_per_day_max']}; "
                  f"decline precision {fmt(pol.get('decline_precision'), 3)}",
                  "- (money figures depend on the cost assumptions in `configs/config.yaml` → `decision_v2`)", ""]

    exp = read_json(art("causal") / "exploration_simulation.json")
    if exp:
        res["exploration"] = exp
        L += ["## Exploration (counterfactual evaluation)", "", "| Candidate block rate | true recall | naive | IPS |", "|---|---|---|---|"]
        for c in exp["candidates"]:
            L.append(f"| {c['candidate_block_rate']} | {c['truth_recall']} | {c['naive_recall']} | {c['ips_recall']} |")
        L += ["", f"Exploration cost: {exp['explored_transactions']} transactions let through.", ""]

    lat = read_json(RESULTS / "latency.json")
    if lat:
        res["latency"] = lat
        L += ["## Latency (end-to-end decision, one transaction at a time)", "",
              "| Stage | p50 ms | p95 ms | p99 ms |", "|---|---|---|---|"]
        for k in ["features", "model", "rules_policy", "total"]:
            if k in lat:
                L.append(f"| {k} | {lat[k]['p50']} | {lat[k]['p95']} | {lat[k]['p99']} |")
        L.append("")

    seq = read_json(art("causal") / "seq_experiment.json")
    if seq:
        res["sequence_experiment"] = seq
        L += ["## Optional GRU sequence score", "",
              f"CV AUC-PR {seq['base_cv_ap']:.4f} → {seq['with_seq_cv_ap']:.4f} (gain {seq['gain']:+.4f}) → **{seq['verdict']}**", ""]

    sim = read_json(ROOT / "artifacts" / "v2" / "sim" / "metrics.json")
    if sim:
        res["simulated_track"] = sim
        L += ["## Simulated world (Track C) — engineering demonstration, NOT real-world accuracy", "",
              f"Val AUC-PR **{fmt(sim['val'].get('auc_pr'))}**, test AUC-PR **{fmt(sim['test'].get('auc_pr'))}** "
              f"(test recall @1% FPR {fmt(sim['test'].get('recall_at_fpr_1pct'))}); features: {sim.get('n_features')}", "",
              "| Fraud scenario (test) | Transactions | Recall @1% FPR |", "|---|---|---|"]
        for sc, v in sim.get("test_by_scenario_at_1pct_fpr", {}).items():
            if not sc.startswith("_"):
                novel = " (NOVEL — never in training)" if sc in (sim.get("novel_scenarios") or {}) else ""
                L.append(f"| {sc}{novel} | {v['n']} | {v['recall_at_1pct_fpr']:.1%} |")
        rings = sim.get("test_by_scenario_at_1pct_fpr", {}).get("_rings")
        if rings:
            L += ["", f"Mule rings detected (≥1 member flagged): **{rings['rings_detected']:.1%}** of "
                      f"{rings['n_rings']}; ring transactions flagged: {rings['ring_transactions_flagged']:.1%}"]
        L.append("")

    fin = art("causal") / "submission_blend.csv"
    if fin.exists():
        L += ["## Final model", "", f"Refit on train+val; Kaggle test predictions: `{fin.relative_to(ROOT)}` "
              f"(+ client-smoothed variant). Serving bundle: `artifacts/v2/causal/bundle_final.json`.", ""]

    L += ["## Step status", "", "| Step | Status | Minutes |", "|---|---|---|"]
    for name, s in state["steps"].items():
        L.append(f"| {name} | {s.get('status')} | {s.get('minutes', '—')} |")
    L += ["", "Logs: `results/logs/`. Full ablation history: `docs/ABLATIONS.md`. Plan: `docs/IMPROVEMENT_PLAN.md`."]

    (RESULTS / "FINAL_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    (RESULTS / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

QUIET_CONSOLE = False


def main():
    global QUIET_CONSOLE
    for stream in (sys.stdout, sys.stderr):              # Windows consoles: never crash on ✓ / ✗
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fast run: fewer trials/epochs, 1 seed, LightGBM-only tuning")
    ap.add_argument("--resume", action="store_true", help="(default) continue from the last checkpoint")
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoints and run every step again")
    ap.add_argument("--low-memory", action="store_true", help="smaller settings for machines with little RAM/GPU memory")
    ap.add_argument("--only", default=None, help="run just this step (after setup/gpu_check)")
    ap.add_argument("--list", action="store_true", help="list steps and exit")
    ap.add_argument("--cpu", action="store_true", help="force CPU (skip the GPU check)")
    ap.add_argument("--require-gpu", action="store_true", help="abort instead of falling back to CPU")
    ap.add_argument("--no-install", action="store_true", help="never pip install anything")
    ap.add_argument("--skip-torch-install", action="store_true")
    ap.add_argument("--trials", type=int, default=40, help="Optuna trials per model")
    ap.add_argument("--full-only", action="store_true", help="skip the raw (real-world) schema track")
    ap.add_argument("--with-transductive", action="store_true", help="also run the Kaggle-style transductive variant")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-sequence", action="store_true")
    ap.add_argument("--skip-sim", action="store_true", help="skip the simulated-world track")
    ap.add_argument("--quiet", action="store_true", help="don't echo step output to the console (still logged)")
    ap.add_argument("--dry-run", action="store_true", help="check setup and print every command without running it")
    ap.add_argument("--no-venv", action="store_true", help="use the current Python instead of creating ./venv")
    args = ap.parse_args()
    if not args.no_venv and not args.list and not _in_project_venv():
        bootstrap_venv(sys.argv[1:])            # never returns
    QUIET_CONSOLE = args.quiet

    RESULTS.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
    resume = not args.fresh
    state = load_state() if resume else {"steps": {}}
    if resume and state.get("steps"):
        say(f"Resuming from checkpoint results/state.json ({sum(1 for v in state['steps'].values() if v.get('status') == 'done')} steps done). Use --fresh to start over.")

    if args.list:
        for name, req, _ in build_steps(args, "gpu"):
            print(f"{name:12s} {'required' if req else 'optional'}")
        return

    say("=" * 70)
    say(f"run_all.py — {'QUICK' if args.quick else 'FULL'} run | python {sys.version.split()[0]} | {ROOT}")
    t_start = time.time()

    # 0 + 1: setup and device
    if args.dry_run:
        args.no_install = True
    try:
        env = step_setup(args)
        env.update({"device": "cpu" if args.cpu else "gpu", "smoke": "not run (--dry-run)"} if args.dry_run
                   else step_gpu_check(args))
    except Exception as e:
        say(f"SETUP FAILED: {e}")
        state["steps"]["setup"] = {"status": f"FAILED: {e}"}; save_state(state); build_report(state)
        sys.exit(1)
    state["env"] = env
    say(f"Environment: {env}")
    if not env.get("torch_cuda") and not args.skip_sequence and env.get("device") == "gpu":
        say("CUDA torch unavailable → skipping the optional GRU sequence step.")
        args.skip_sequence = True
    save_state(state)

    steps = build_steps(args, env["device"])
    if args.only:
        steps = [s for s in steps if s[0] == args.only]
        if not steps:
            sys.exit(f"unknown step {args.only}; see --list")

    for name, required, cmds in steps:
        prev = state["steps"].get(name, {})
        if resume and prev.get("status") == "done":
            say(f"✓ {name}: already done (checkpoint) — skipping"); continue
        say(f"▶ {name} ({'required' if required else 'optional'})")
        if args.dry_run:
            for cmd in cmds:
                say("    would run: " + " ".join(str(c).replace(PY, "python") for c in cmd))
            continue
        t0 = time.time()
        failed = None
        done_cmds = state.setdefault("commands", {})
        for i, cmd in enumerate(cmds):
            key = " ".join(str(c) for c in cmd[1:])            # checkpoint per command
            if resume and done_cmds.get(key) == "done":
                say(f"    ✓ already done (checkpoint): {key}"); continue
            say(f"    memory before: {memory_status()}")
            rc = run_cmd(name, cmd)                           # separate process → memory freed on exit
            if rc != 0:
                oom = looks_like_oom(rc, LOGS / f"{name}.log")
                failed = (f"command {i + 1}/{len(cmds)} exited {rc}"
                          + (" — looks like OUT OF MEMORY: re-run `python run_all.py` to resume, "
                             "or add --low-memory" if oom else ""))
                done_cmds[key] = "failed"; save_state(state)
                break
            done_cmds[key] = "done"; save_state(state)       # checkpoint immediately
        minutes = round((time.time() - t0) / 60, 1)
        state["steps"][name] = {"status": "done" if not failed else f"FAILED ({failed})", "minutes": minutes}
        save_state(state)
        if failed:
            say(f"✗ {name} FAILED after {minutes} min — {failed}. Log: results/logs/{name}.log")
            if required:
                build_report(state)
                say("Required step failed → stopping. Fix the issue, then run `python run_all.py` again "
                    "(it resumes from this command).")
                sys.exit(1)
        else:
            say(f"✓ {name} done in {minutes} min")

    if args.dry_run:
        say("Dry run finished — nothing was executed.")
        return
    build_report(state)
    total = round((time.time() - t_start) / 60, 1)
    say(f"ALL DONE in {total} min → results/FINAL_REPORT.md")
    print("\n" + (RESULTS / "FINAL_REPORT.md").read_text(encoding="utf-8")[:4000])


if __name__ == "__main__":
    main()
