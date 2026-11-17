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
 13  sequence     Phase D sequence-embedding keep-rule test (needs CUDA torch on GPU runs; skipped if absent)
 14  final        refit on train+val → Kaggle test predictions (submission CSV)
 15  sim_generate simulated payments world (real-world fields IEEE lacks) → data/synthetic/v2_dense
 16  sim_train    train + evaluate on it: per-scenario recall, novel scenario, ring detection
 16b closed_loop  closed-loop experiments: selection bias/exploration, retraining, cost actions, novelty
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

ROBUSTNESS (lessons from the first GPU run: 11 GPU-hours lost to a shutdown crash)
  * SUCCESS SENTINEL. Every project script is launched through src/utils/run_guard.py, which
    writes a sentinel file after the script's main() returned, flushes all logs and ends the
    process with os._exit(0) (no native-library teardown). A command whose sentinel exists
    counts as done even if the exit code is non-zero (warning in the log), e.g. the
    0xC0000005 access violation that Python 3.14 + CUDA/OpenCL libraries raise at exit.
  * "OUT OF MEMORY" is only reported when the log really shows a memory error.
  * LOCK FILE results/run_all.lock: a second run_all refuses to start while another live one
    holds it (stale locks of dead processes are taken over). Use --force-lock to override.
  * PYTHON VERSION: 3.11-3.13 only. The bootstrap picks such an interpreter (py launcher /
    python3.13 ...) and refuses 3.14 unless --allow-unsupported-python is given.
  * CatBoost is OFF by default (it got blend weight 0 and took 2.8 h of tuning in the first
    run); --with-catboost adds it to tuning and training.
  * Label policy: --label-median-days / --maturity-days / --train-gap-days re-run the whole
    pipeline under another chargeback-delay assumption (30/60/90); --tag-suffix keeps the
    outputs of different policies apart (e.g. --label-median-days 60 --tag-suffix _d60).
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
sys.path.insert(0, str(ROOT))
PY = sys.executable
from src.utils.run_guard import (                      # stdlib-only helpers (also usable before the venv exists)
    SENTINEL_ENV, acquire_lock, choose_python, classify_outcome, py_supported, release_lock,
)
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"
STATE = RESULTS / "state.json"
LOCK = RESULTS / "run_all.lock"
PROTOCOL = "A1-label-policy-v1"          # bump when feature/training semantics change: invalidates old checkpoints
SENTINELS = RESULTS / ".sentinels"
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


def _py_version_of(exe: str):
    try:
        out = subprocess.run([exe, "-c", "import sys;print(sys.version_info[0], sys.version_info[1])"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        return int(out[0]), int(out[1])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def find_base_python():
    """Interpreter used to create the venv: Python 3.11-3.13 (newest first).

    3.14 + native GPU libs crashes at interpreter shutdown (0xC0000005), so it is never
    picked automatically. Candidates: the running interpreter, the Windows `py` launcher,
    python3.13 / python3.12 / python3.11 on PATH.
    """
    cands = [(sys.executable, tuple(sys.version_info[:2]))]
    for minor in (13, 12, 11):
        found = []
        if os.name == "nt" and shutil.which("py"):
            try:
                out = subprocess.run(["py", f"-3.{minor}", "-c", "import sys;print(sys.executable)"],
                                     capture_output=True, text=True, timeout=30)
                if out.returncode == 0 and out.stdout.strip():
                    found.append(out.stdout.strip().splitlines()[0])
            except (OSError, subprocess.TimeoutExpired):
                pass
        exe = shutil.which(f"python3.{minor}")
        if exe:
            found.append(exe)
        for f in found:
            v = _py_version_of(f)
            if v:
                cands.append((f, v))
    return choose_python(cands)


def _unsupported_python_message(version) -> str:
    return (f"Python {version[0]}.{version[1]} is not supported (use 3.11-3.13). On Python 3.14 the native GPU "
            "libraries (LightGBM/XGBoost/CatBoost/torch) crash at interpreter shutdown with 0xC0000005 and "
            "the first GPU run lost ~11 hours that way. Install Python 3.13 (https://www.python.org/downloads/) "
            "and run again, or pass --allow-unsupported-python at your own risk.")


def bootstrap_venv(argv) -> None:
    """Create ./venv if needed, install CUDA torch + requirements-gpu.txt, then re-exec in the venv."""
    allow = "--allow-unsupported-python" in argv
    base = find_base_python()
    if base is None:
        msg = _unsupported_python_message(sys.version_info[:2])
        if not allow:
            sys.exit("[bootstrap] " + msg)
        _boot_say("WARNING: " + msg)
        base = (sys.executable, tuple(sys.version_info[:2]))
    if sys.version_info < (3, 11) and not allow:
        sys.exit(f"Python {sys.version.split()[0]} is too old: install Python 3.11-3.13 and run again.")
    if VENV_PY.exists():
        vv = _py_version_of(str(VENV_PY))
        if vv and not py_supported(vv) and not allow:
            sys.exit(f"[bootstrap] the existing venv/ uses Python {vv[0]}.{vv[1]}. " + _unsupported_python_message(vv)
                     + " Delete the venv/ folder so it is re-created with a supported interpreter.")
    if not VENV_PY.exists():
        _boot_say(f"creating virtual environment in {VENV} with {base[0]} (Python {base[1][0]}.{base[1][1]}) ...")
        subprocess.run([base[0], "-m", "venv", str(VENV)], check=True)
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


def run_cmd(step: str, cmd: list, log_name: str | None = None, guard: bool = False,
            env_extra: dict | None = None) -> int:
    """Run a command, streaming output to the console and to results/logs/<step>.log.

    guard=True launches a project script through src/utils/run_guard.py (success sentinel +
    hard exit, see the module docstring). Returns the exit code; ``LAST_RUN`` records the
    sentinel and the log text of this command for ``classify_outcome``.
    """
    log = LOGS / f"{log_name or step}.log"
    cmd = [str(c) for c in cmd]
    sentinel = None
    if guard and len(cmd) > 1 and cmd[1].endswith(".py"):
        SENTINELS.mkdir(parents=True, exist_ok=True)
        sentinel = SENTINELS / f"{step}_{abs(hash(' '.join(cmd))) % 10**8}_{int(time.time())}.ok"
        cmd = [cmd[0], "-m", "src.utils.run_guard", *cmd[1:]]
    say(f"  $ {' '.join(cmd)}   (log: {log.relative_to(ROOT)})")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", **(env_extra or {})}
    if sentinel:
        env[SENTINEL_ENV] = str(sentinel)
    else:
        env.pop(SENTINEL_ENV, None)
    start = log.stat().st_size if log.exists() else 0
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== {dt.datetime.now()} :: {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env, cwd=ROOT)
        for line in proc.stdout:
            f.write(line)
            if not QUIET_CONSOLE:
                sys.stdout.write("    " + line)
        rc = proc.wait()
    try:
        with open(log, "rb") as f:
            f.seek(start)
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        text = ""
    ok = bool(sentinel and sentinel.exists())
    LAST_RUN.update({"sentinel": ok, "log": text})
    if sentinel and sentinel.exists():
        try:
            sentinel.unlink()
        except OSError:
            pass
    return rc


LAST_RUN: dict = {"sentinel": False, "log": ""}


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
    """Kept for callers/tests: OOM only if the log shows a memory error (see run_guard)."""
    from src.utils.run_guard import looks_like_oom as _l
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    return _l(rc, text)


def read_json(path) -> dict | None:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except json.JSONDecodeError:
        return None


def art(tag: str) -> Path:
    return ROOT / "artifacts" / "v2" / (tag + os.environ.get("FRAUD_TAG_SUFFIX", ""))


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
    rc = run_cmd("gpu_check", [PY, "scripts/gpu_smoke_test.py", "--allow-lgbm-cpu", "--torch-optional"], guard=True)
    if classify_outcome(rc, LAST_RUN["sentinel"], LAST_RUN["log"])[0] == "done":
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

def sim_world_ok(d: Path) -> bool:
    """v2_dense world present and generated with the realistic_dense preset (metadata check)."""
    try:
        m = json.loads((d / "metadata.json").read_text())
        c = m["config"]
        return (c.get("preset") == "realistic_dense" and m.get("generator") == "sim-v2"
                and (d / "transactions.parquet").exists() and (d / "events.parquet").exists())
    except Exception:
        return False


def build_steps(args, device: str) -> list:
    dev = ["--device", device]
    quick = args.quick or args.low_memory
    # CatBoost got blend weight 0 and cost 2.8 h of tuning in the first GPU run: opt-in only
    tuning_models = ["lgbm"] if quick else ["lgbm", "xgb"] + (["catboost"] if args.with_catboost else [])
    mdl = ["--models", *(["lgbm", "xgb"] + (["catboost"] if args.with_catboost else []))]
    pol = []                                   # label-policy overrides for the feature build
    if args.label_median_days is not None:
        pol += ["--label-median-days", str(args.label_median_days)]
    if args.maturity_days is not None:
        pol += ["--maturity-days", str(args.maturity_days)]
    gap = ["--train-gap-days", str(args.train_gap_days)] if args.train_gap_days is not None else []
    trials = "15" if quick else str(args.trials)
    schemas = ["full", "raw"] if not args.full_only else ["full"]
    S = []   # (name, required, [commands])
    S.append(("preprocess", True, [[PY, "scripts/run_preprocessing.py"]]))
    S.append(("features", True, [[PY, "scripts/run_feature_engineering.py", "--schema", s, *pol] for s in schemas]
              + ([[PY, "scripts/run_feature_engineering.py", "--agg-mode", "transductive", *pol]] if args.with_transductive else [])))
    S.append(("selection", True, [[PY, "scripts/run_feature_selection.py", "--two-windows", "--schema", s] for s in schemas]))
    if not args.skip_ablation:
        abl_schemas = ["full"] if quick else schemas
        abl = ["--seeds", "1" if quick else str(args.ablation_seeds)]
        S.append(("ablation", False, [[PY, "scripts/run_ablation.py", "--schema", s, *dev, *abl] for s in abl_schemas]))
    S.append(("tuning", False, [[PY, "scripts/run_tuning.py", "--model", m, "--trials", trials, *dev] for m in tuning_models]))
    seeds = ["--n-seeds", "1"] if quick else []
    S.append(("training", True, [[PY, "scripts/run_training.py", "--use-tuned", *mdl, *gap, *seeds, *dev]]
              + [[PY, "scripts/run_training.py", "--schema", s, *mdl, *gap, *seeds, *dev] for s in schemas if s != "full"]
              + ([[PY, "scripts/run_training.py", "--agg-mode", "transductive", *mdl, *gap, *seeds, *dev]] if args.with_transductive else [])))
    S.append(("heldout", False, [[PY, "scripts/run_heldout_pattern_test.py", *dev]]))
    S.append(("novelty", False, [[PY, "scripts/run_novelty_monitoring.py"]]))
    S.append(("policy", False, [[PY, "scripts/run_policy_simulation.py", "--schema", s] for s in schemas]))
    S.append(("exploration", False, [[PY, "scripts/run_exploration_simulation.py"]]))
    if not args.skip_latency:
        S.append(("latency", False, [[PY, "scripts/benchmark_latency.py", "--n", "500" if quick else "2000",
                                      "--out", "results/latency.json"]]))
    if not args.skip_sequence:
        S.append(("sequence", False, [[PY, "scripts/run_sequence_experiment.py", *dev]
                                      + (["--epochs", "3", "--seeds", "1"] if quick else ["--seeds", "3"])]))
    # --final reuses the validation run's CV (rounds + OOF) when its fingerprint is identical
    S.append(("final", False, [[PY, "scripts/run_training.py", "--final", "--use-tuned", *mdl, *gap, *seeds, *dev]]))
    if not args.skip_sim:
        sim_dir = ROOT / "data" / "synthetic" / "v2_dense"
        if sim_world_ok(sim_dir):
            print(f"[sim] {sim_dir} exists and its metadata matches realistic_dense: skipping generation")
        else:
            sim_gen = [PY, "scripts/run_simulator.py", "--preset", "realistic_dense", "--out", "data/synthetic/v2_dense"]
            if quick:
                sim_gen += ["--no-validate", "--no-ceiling"]
            S.append(("sim_generate", False, [sim_gen]))
        S.append(("sim_train", False, [[PY, "scripts/run_sim_track.py", "--data", "data/synthetic/v2_dense", *dev]
                                       + (["--quick", "--max-customers", "15000"] if quick else [])]))
        S.append(("closed_loop", False, [[PY, "scripts/run_closed_loop.py"]
                                         + (["--quick"] if quick else ["--size", "full", "--seeds", "1", "2", "3", "4", "5"])]))
    return S


# ─────────────────────────────────────────────────────────────────────────────
# report
# ─────────────────────────────────────────────────────────────────────────────

def _nmean(xs):
    v = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return sum(v) / len(v) if v else None


def fmt(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def model_table(m: dict) -> list:
    rows = ["| Model | CV AUC-PR (mean +- SE) | Val AUC-PR | Val ROC-AUC | Recall @1% FPR | VDR @1% FPR | Precision top 1% |",
            "|---|---|---|---|---|---|---|"]
    for name, r in (m.get("models") or {}).items():
        v = r.get("val", {})
        rows.append(f"| {name} | {fmt(r.get('cv_ap_mean'))} +- {fmt(r.get('cv_ap_se'))} | {fmt(v.get('auc_pr'))} | "
                    f"{fmt(v.get('auc_roc'))} | {fmt(v.get('recall_at_fpr_1pct'))} | {fmt(v.get('vdr_at_fpr_1pct'))} | "
                    f"{fmt(v.get('precision_top_1pct'))} |")
    b = m.get("blend_val") or {}
    rows.append(f"| **blend (ranking score)** | {fmt(m.get('blend_cv_ap_mean'))} +- {fmt(m.get('blend_cv_ap_se'))} | "
                f"**{fmt(b.get('auc_pr'))}** | {fmt(b.get('auc_roc'))} | {fmt(b.get('recall_at_fpr_1pct'))} | "
                f"{fmt(b.get('vdr_at_fpr_1pct'))} | {fmt(b.get('precision_top_1pct'))} |")
    return rows


def slice_table(m: dict) -> list:
    sl = m.get("blend_val_slices") or {}
    if not sl:
        return []
    rows = ["| Slice | Rows | Fraud rows | Base rate | AUC-PR | AP / base | Recall @1% FPR | VDR @1% FPR | CP@k daily |",
            "|---|---|---|---|---|---|---|---|---|"]
    names = {"all": "all validation rows", "client_seen": "client seen in training",
             "cold_start": "COLD START (client never seen)", "uid_unbuildable": "client id unbuildable (addr1/D1 missing)",
             "copied_removed": "COPIED POSITIVES REMOVED"}
    for k, label in names.items():
        r = sl.get(k)
        if not r:
            continue
        if r.get("skipped"):
            rows.append(f"| {label} | {r['n']} | {r['n_fraud']} | - | skipped ({r['skipped']}) | | | | |")
            continue
        cp = next((v for kk, v in r.items() if kk.startswith("cp_at_") and kk.endswith("_daily")), None)
        rows.append(f"| {label} | {r['n']} | {r['n_fraud']} | {fmt(r.get('base_rate'))} | {fmt(r.get('auc_pr'))} | "
                    f"{fmt(r.get('ap_over_base_rate'), 1)} | {fmt(r.get('recall_at_fpr_1pct'))} | "
                    f"{fmt(r.get('vdr_at_fpr_1pct'))} | {fmt(cp)} |")
    return rows


def build_report(state: dict) -> None:
    res = {"generated": dt.datetime.now().isoformat(), "environment": state.get("env", {}),
           "old_baseline": OLD_BASELINE, "steps": state["steps"]}
    L = ["# Fraud detection v2 — final results", "",
         f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Device: **{state.get('env', {}).get('device', '?')}** "
         f"({state.get('env', {}).get('gpu', '?')}). Smoke test: {state.get('env', {}).get('smoke', '?')}.", ""]
    if state.get("warnings"):
        L += ["Runner warnings (step finished its work but the process crashed at shutdown; ignored):"] + \
             [f"- {w}" for w in state["warnings"][-10:]] + [""]

    full = read_json(art("causal") / "metrics_val.json")
    raw = read_json(art("causal_raw") / "metrics_val.json")
    trans = read_json(art("transductive") / "metrics_val.json")
    res["metrics_full"], res["metrics_raw"], res["metrics_transductive"] = full, raw, trans

    pol = next((m.get("label_policy") for m in (full, raw, trans) if m and m.get("label_policy")), None)
    if pol:
        res["label_policy"] = pol
        L += ["## Label policy (identical for features, training targets and evaluation)", "",
              f"- A fraud label becomes known `d` days after the transaction, `d` ~ lognormal(median {pol['delay_median_days']:g} d, "
              f"sigma {pol['delay_sigma']:g}), capped at {pol['delay_cap_days']:g} d, deterministic per TransactionID.",
              f"- A transaction without a known fraud label is treated as legit once it is {pol['maturity_days']:g} d old.",
              f"- Training targets are the labels known at each cutoff; the last {pol['train_gap_days']:g} d before the cutoff are "
              "excluded (train | gap | val). Validation/test are scored against the TRUE labels.",
              "- Early stopping uses the first half of each validation window; only the second half is scored.",
              "- Numbers are therefore lower than the first run (old protocol: labels known after exactly 30 d, early stopping "
              "on the scored rows) -- that is the point.", ""]

    L += ["## Headline (validation holdout = days 131-182, never used for training or tuning)", "",
          "| Model | AUC-PR (all rows) | Base rate | AUC-PR cold-start clients | AUC-PR copied positives removed | Recall @1% FPR | VDR @1% FPR |",
          "|---|---|---|---|---|---|---|"]
    for label, m in [("v2 full schema (all IEEE-CIS columns)", full), ("v2 raw schema (real-world fields only — served live)", raw),
                     ("v2 transductive (Kaggle-style, optimistic)", trans)]:
        if m and m.get("blend_val"):
            b, sl = m["blend_val"], m.get("blend_val_slices") or {}
            L.append(f"| {label} | **{fmt(b.get('auc_pr'))}** | {fmt(b.get('base_rate'))} | "
                     f"**{fmt((sl.get('cold_start') or {}).get('auc_pr'))}** | **{fmt((sl.get('copied_removed') or {}).get('auc_pr'))}** | "
                     f"{fmt(b.get('recall_at_fpr_1pct'))} | {fmt(b.get('vdr_at_fpr_1pct'))} |")
    L += ["", f"Old baselines (different, optimistic protocol: labels known after 30 d, early stopping on the scored rows): "
          f"{OLD_BASELINE['val_auc_pr_notebook02_days131_182']} (notebook 02, same val window), "
          f"{OLD_BASELINE['val_auc_pr_fold_average_days96_130']} (old scripted pipeline, days 96-130). "
          "Not comparable with the numbers above.", "",
          "Cold-start = clients never seen in training (what transfers to new customers in production). "
          "Copied positives removed = fraud rows that come after the same client's first fraud are dropped "
          "(IEEE propagated chargeback labels to later rows: remembering the client is not detection).", ""]

    for label, m in [("Full schema", full), ("Raw (real-world) schema", raw)]:
        if not m:
            continue
        L += [f"## Models — {label}", ""] + model_table(m)
        cal = m.get("blend_val_calibration") or {}
        sm = m.get("blend_val_client_smoothed") or {}
        b = m.get("blend_val") or {}
        cp = next((v for kk, v in b.items() if kk.startswith("cp_at_") and kk.endswith("_daily")), None)
        L += ["", f"Blend weights (chosen on out-of-fold scoring windows): `{m.get('blend_weights')}`  ",
              "AUC-PR is computed on the uncalibrated blend score (ranking); the calibrated probability has the same order.  ",
              f"Calibration (probabilities): Brier {fmt(cal.get('brier'))}, ECE {fmt(cal.get('ece'))}  ",
              f"False-decline rate at the top-1% review budget: {fmt(b.get('false_decline_rate_top1pct'), 5)}; "
              f"flag precision {fmt(b.get('flag_precision_top1pct'))}; card/client precision@k daily: {fmt(cp)}  ",
              f"Client-smoothed blend AUC-PR: {fmt(sm.get('auc_pr'))}  ",
              f"Features used: {m.get('n_features')}", ""]
        folds = m.get("folds") or []
        if folds:
            ml = list((m.get("models") or {}))
            L += ["Rolling folds (scored half of each validation window):", "",
                  "| Fold | Days | Train rows (known fraud) | Scored rows | Base rate | " + " | ".join(f"{n} AP (AP/base)" for n in ml) + " |",
                  "|---|---|---|---|---|" + "---|" * len(ml)]
            for i, f in enumerate(folds):
                cells = []
                for n in ml:
                    r = m["models"][n]
                    cells.append(f"{fmt(r['cv_ap_folds'][i], 3)} ({fmt(r['cv_ap_over_base_folds'][i], 1)})"
                                 if i < len(r.get("cv_ap_folds", [])) else "-")
                L.append(f"| {f['fold']} | {f['val_days'][0]}-{f['val_days'][1]} | {f['n_train']} ({f['train_pos_known']}) | "
                         f"{f['n_score']} | {f['score_base_rate']:.4f} | " + " | ".join(cells) + " |")
            L.append("")
        st = slice_table(m)
        if st:
            L += [f"Slices ({label}):", ""] + st + [""]

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
                  "Paired per-fold deltas averaged over several seeds; KEEP needs gain > 2 x SE and >= 0.005.", "",
                  "| Family | # cols | CV AUC-PR without it | Gain from family (+- SE) | Folds helped | Verdict |", "|---|---|---|---|---|---|"]
            for _, r in ab.iterrows():
                se = f" +- {r['delta_se']:.4f}" if "delta_se" in ab.columns and r["delta_se"] == r["delta_se"] else ""
                L.append(f"| {r['family']} | {r['n_removed']} | {r['ap_mean']:.4f} | {r['delta_if_removed']:+.4f}{se} | "
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
        L += ["## Sequence embeddings (Phase D: self-supervised GRU, causal features added to the GBDT)", "",
              f"{seq.get('n_folds')} rolling folds x {seq.get('seeds')} seeds; deltas paired per fold vs base (mean +/- SE).", "",
              "| Arm | AUC-PR delta | Cold-start AUC-PR delta | Copied-removed delta | VDR@1% FPR delta | Verdict |",
              "|---|---|---|---|---|---|"]
        for arm, d in (seq.get("arms") or {}).items():
            if arm == "base":
                continue
            cell = lambda k: (f"{d[k]['delta']:+.4f} +/- {d[k]['se']:.4f}" if k in d and d[k].get("delta") is not None else "n/a")
            L.append(f"| {arm} | {cell('ap_all')} | {cell('ap_cold_start')} | {cell('ap_copied_removed')} | "
                     f"{cell('vdr_1pct')} | **{d.get('verdict')}** |")
        L.append("")

    sim = read_json(ROOT / "artifacts" / "v2" / "sim" / "metrics.json")
    if sim:
        res["simulated_track"] = sim
        te = sim["test"]
        tt, tr = te.get("true", {}), te.get("reported", {})
        L += ["## Simulated world (Track C, v2_dense) - engineering demonstration, NOT real-world accuracy", "",
              f"Causal features: {sim.get('n_features')} (+{sim.get('n_burst_features')} burst). Targets as known at the "
              f"cutoff: {sim.get('label_semantics')}.", "",
              "| Test metric | vs TRUE fraud (is_fraud_true) | vs REPORTED labels (business view) |", "|---|---|---|"]
        for lab, key in [("AUC-PR", "auc_pr"), ("Recall @1% FPR", "recall_at_fpr_1pct"),
                         ("Value detection @1% FPR", "vdr_at_fpr_1pct"), ("CP@k (daily)", "cp_at_100_daily"),
                         ("False-decline rate (top 1%)", "false_decline_rate_top1pct")]:
            L.append(f"| {lab} | {fmt(tt.get(key))} | {fmt(tr.get(key))} |")
        L += ["", "| Fraud scenario (test) | Rows | Flagged at 1% FPR |", "|---|---|---|"]
        for sc, v in te.get("by_scenario", {}).items():
            novel = " (NOVEL - test only)" if sc in (te.get("novel") or {}) else ""
            L.append(f"| {sc}{novel} | {v['n']} | {v['flagged_rate']:.1%} |")
        for k, v in (te.get("slices") or {}).items():
            L.append(f"| slice: {k} | {v.get('n')} | recall {fmt(v.get('recall_at_op'))} (AP {fmt(v.get('auc_pr'))}) |")
        ct = sim.get("card_testing_model")
        if ct:
            L += ["", "Dedicated card-testing model (burst features) vs general model, card_testing recall:", "",
                  "| Model | Recall @0.5% FPR | Recall @1% FPR | Overall AUC-PR |", "|---|---|---|---|"]
            for k in ("general", "specialist", "routed_max"):
                L.append(f"| {k} | {fmt(ct[k].get('recall_at_fpr_0.5pct'))} | {fmt(ct[k].get('recall_at_fpr_1pct'))} | "
                         f"{fmt(ct[k].get('auc_pr_overall_true'))} |")
        rings = te.get("rings")
        if rings:
            L += ["", f"Mule rings detected (at least one member flagged): **{rings['rings_detected']:.1%}** of {rings['n_rings']}"]
        L.append("")

    cl = read_json(ROOT / "artifacts" / "v2" / "closed_loop" / "metrics.json")
    if cl:
        res["closed_loop"] = {k: v for k, v in cl.items() if k != "exp4_novelty"}
        L += ["## Closed-loop experiments (simulator; the policy only learns from labels it was delivered)", "",
              f"World size {cl.get('meta', {}).get('size')}, seeds {cl.get('meta', {}).get('seeds')}. Final-model "
              "quality is scored on a fixed held-out stream with true labels.", ""]
        e1 = (cl.get("exp1_selection_bias") or {}).get("summary") or {}
        if e1:
            L += ["**1. Selection bias vs exploration**", "", "| Arm | AUC-PR | Recall @1% FPR | Fraud loss $ | False declines |",
                  "|---|---|---|---|---|"]
            for arm, d in e1.items():
                fm, sm = d.get("final_model") or {}, d.get("summary") or {}
                L.append(f"| {arm} | {fmt(fm.get('auc_pr'))} | {fmt(fm.get('recall_at_1pct_fpr'))} | "
                         f"{fmt(sm.get('fraud_loss_usd'), 0)} | {fmt(sm.get('false_declines'), 0)} |")
            L.append("")
        e2 = (cl.get("exp2_retraining") or {}).get("summary") or {}
        if e2:
            L += ["**2. Retraining strategy (adaptive attackers)**", "", "| Arm | Retrains | AUC-PR | Recall @1% FPR | Fraud loss $ |",
                  "|---|---|---|---|---|"]
            for arm, d in e2.items():
                fm, sm = d.get("final_model") or {}, d.get("summary") or {}
                L.append(f"| {arm} | {fmt(sm.get('n_retrains'), 1)} | {fmt(fm.get('auc_pr'))} | "
                         f"{fmt(fm.get('recall_at_1pct_fpr'))} | {fmt(sm.get('fraud_loss_usd'), 0)} |")
            L.append("")
        e3 = (cl.get("exp3_cost") or {}).get("summary") or {}
        if e3:
            L += ["**3. Decision rule**", "", "| Arm | Net cost $ | Fraud loss $ | False declines | 3DS rate | Review rate |",
                  "|---|---|---|---|---|---|"]
            for arm, d in e3.items():
                sm = d.get("summary") or {}
                L.append(f"| {arm} | {fmt(sm.get('net_cost_usd'), 0)} | {fmt(sm.get('fraud_loss_usd'), 0)} | "
                         f"{fmt(sm.get('false_declines'), 0)} | {fmt(sm.get('stepup_rate'))} | {fmt(sm.get('review_rate'))} |")
            L.append("")
        runs = (cl.get("exp4_novelty") or {}).get("runs") or []
        if runs:
            res["closed_loop"]["exp4_novelty_runs"] = runs
            agg = {}
            for r in runs:
                a = agg.setdefault(r["arm"], [0, 0, [], []])
                a[0] += r.get("n_detected", 0); a[1] += r.get("n_incidents", 0)
                a[2].append(r.get("false_alarm_rate_per_week_pre_test", float("nan")))
                a[3].append((r.get("supervised") or {}).get("bust_out_recall_at_1pct_fpr", float("nan")))
            L += ["**4. Novel scenario (bust_out, test period only)**", "",
                  "| Arm | Incidents detected by novelty layer | False alarms / week (pre-test) | Supervised bust_out recall @1% FPR |",
                  "|---|---|---|---|"]
            for arm, (d, n, fa, sr) in agg.items():
                L.append(f"| {arm} | {d}/{n} | {fmt(_nmean(fa))} | {fmt(_nmean(sr))} |")
            L.append("")

    fin = art("causal") / "submission_blend.csv"
    if fin.exists():
        L += ["## Final model", "", f"Refit on train+val; Kaggle test predictions: `{fin.relative_to(ROOT)}` "
              f"(+ client-smoothed variant). Serving bundle: `artifacts/v2/causal/bundle_final.json`.", ""]

    L += ["## Step status", "", "| Step | Status | Minutes |", "|---|---|---|"]
    for name, s in state["steps"].items():
        L.append(f"| {name} | {s.get('status')} | {s.get('minutes', '—')} |")
    L += ["", "Logs: `results/logs/`. Full ablation history: `docs/ABLATIONS.md`. Plan: `docs/IMPROVEMENT_PLAN.md`."]

    sfx = os.environ.get("FRAUD_TAG_SUFFIX", "")
    (RESULTS / f"FINAL_REPORT{sfx}.md").write_text("\n".join(L), encoding="utf-8")
    (RESULTS / f"results{sfx}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

QUIET_CONSOLE = False


def main():
    global QUIET_CONSOLE, STATE
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
    ap.add_argument("--with-catboost", action="store_true", help="also tune/train CatBoost (off by default: blend weight 0, 2.8 h)")
    ap.add_argument("--ablation-seeds", type=int, default=3, help="seeds per ablation arm (paired, averaged)")
    ap.add_argument("--label-median-days", type=float, default=None, help="label policy: median chargeback delay (default config: 35)")
    ap.add_argument("--maturity-days", type=float, default=None, help="label policy: days until an unreported txn counts as legit (default 90)")
    ap.add_argument("--train-gap-days", type=float, default=None, help="label policy: rows this close to the cutoff are not trained on")
    ap.add_argument("--tag-suffix", default="", help="suffix for feature/artifact folders, e.g. _d60, to keep policy variants apart")
    ap.add_argument("--allow-unsupported-python", action="store_true", help="allow Python 3.14+ (native GPU libs may crash at exit)")
    ap.add_argument("--force-lock", action="store_true", help="ignore results/run_all.lock (another run is active!)")
    ap.add_argument("--with-transductive", action="store_true", help="also run the Kaggle-style transductive variant")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-sequence", action="store_true")
    ap.add_argument("--skip-sim", action="store_true", help="skip the simulated-world track")
    ap.add_argument("--quiet", action="store_true", help="don't echo step output to the console (still logged)")
    ap.add_argument("--dry-run", action="store_true", help="check setup and print every command without running it")
    ap.add_argument("--no-venv", action="store_true", help="use the current Python instead of creating ./venv")
    args = ap.parse_args()
    if args.tag_suffix:
        os.environ["FRAUD_TAG_SUFFIX"] = args.tag_suffix          # inherited by every step (dataset_tag)
        STATE = RESULTS / f"state{args.tag_suffix}.json"
    if not args.no_venv and not args.list and not _in_project_venv():
        bootstrap_venv(sys.argv[1:])            # never returns
    if not args.list and not py_supported(tuple(sys.version_info[:2])):
        msg = _unsupported_python_message(sys.version_info[:2])
        if not args.allow_unsupported_python:
            sys.exit("[run_all] " + msg)
        print("[run_all] WARNING: " + msg, flush=True)
    QUIET_CONSOLE = args.quiet

    RESULTS.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
    if not args.list and not args.dry_run and not args.force_lock:
        ok, holder = acquire_lock(LOCK, {"cmd": " ".join(sys.argv)})
        if not ok:
            sys.exit(f"[run_all] another run_all is active (pid {holder.get('pid')}, started {holder.get('started')}): "
                     f"{holder.get('cmd')}. Two runs would overwrite each other's models and state. "
                     f"Wait for it, stop it, or pass --force-lock if you are sure it is dead.")
        import atexit
        atexit.register(release_lock, LOCK)
    resume = not args.fresh
    state = load_state() if resume else {"steps": {}}
    if state.get("steps") and state.get("protocol") != PROTOCOL:
        say(f"Checkpoint results/{STATE.name} was written by an older protocol ({state.get('protocol')!r}, before the "
            f"label policy / rolling-fold fixes): its features and models are NOT comparable, so every step is re-run.")
        state = {"steps": {}}
    state["protocol"] = PROTOCOL
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
            rc = run_cmd(name, cmd, guard=True)               # separate process → memory freed on exit
            status, warn = classify_outcome(rc, LAST_RUN["sentinel"], LAST_RUN["log"])
            if warn:
                say(f"    WARNING: {warn}")
                state.setdefault("warnings", []).append(f"{name}: {key}: {warn}")
            if status != "done":
                failed = (f"command {i + 1}/{len(cmds)} exited {rc}"
                          + (" — the log shows an OUT OF MEMORY error: re-run `python run_all.py` to resume, "
                             "or add --low-memory" if status == "oom" else ""))
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
    print("\n" + (RESULTS / f"FINAL_REPORT{os.environ.get('FRAUD_TAG_SUFFIX', '')}.md").read_text(encoding="utf-8")[:4000])


if __name__ == "__main__":
    main()
