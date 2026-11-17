"""Runner robustness: success sentinel, outcome classification, lock file, interpreter choice."""
import json
import os
import subprocess
import sys
from pathlib import Path

from src.utils import run_guard as rg

ROOT = Path(__file__).resolve().parent.parent


# ── decision function ─────────────────────────────────────────────────────────

def test_sentinel_beats_a_shutdown_crash_exit_code():
    st, warn = rg.classify_outcome(3221225477, sentinel_exists=True, log_text="Report saved")
    assert st == "done" and "0xC0000005" in warn


def test_clean_exit_and_unguarded_success():
    assert rg.classify_outcome(0, True, "") == ("done", None)
    assert rg.classify_outcome(0, False, "") == ("done", None)         # e.g. pip, unguarded command


def test_crash_without_sentinel_is_failed_not_oom():
    # access violation after "Report saved" but no sentinel: NOT out-of-memory
    assert rg.classify_outcome(3221225477, False, "INFO Report saved to x")[0] == "failed"
    assert rg.classify_outcome(1, False, "Traceback (most recent call last):\nImportError: x")[0] == "failed"


def test_oom_only_when_the_log_shows_a_memory_error():
    assert rg.classify_outcome(1, False, "numpy._core._exceptions._ArrayMemoryError: Unable to allocate 2.1 GiB")[0] == "oom"
    assert rg.classify_outcome(1, False, "MemoryError")[0] == "oom"
    assert rg.classify_outcome(1, False, "lightgbm: CUDA out of memory")[0] == "oom"
    assert rg.classify_outcome(137, False, "")[0] == "oom"              # killed by the kernel OOM killer
    assert rg.classify_outcome(-1073741819, False, "")[0] == "failed"   # 0xC0000005 alone is not memory


# ── lock file ─────────────────────────────────────────────────────────────────

def test_lock_status_and_stale_takeover(tmp_path):
    lock = tmp_path / "run_all.lock"
    assert rg.lock_status(None) == "free"
    assert rg.lock_status({"pid": 123}, alive=lambda p: True) == "held"
    assert rg.lock_status({"pid": 123}, alive=lambda p: False) == "stale"
    assert rg.lock_status({}, alive=lambda p: True) == "stale"                       # unreadable lock
    assert rg.lock_status({"pid": os.getpid()}, alive=lambda p: True, my_pid=os.getpid()) == "own"

    ok, _ = rg.acquire_lock(lock, {"cmd": "a"}, alive=lambda p: True)
    assert ok and json.loads(lock.read_text())["pid"] == os.getpid()
    lock.write_text(json.dumps({"pid": 99999999, "started": "x", "cmd": "other"}))
    ok, holder = rg.acquire_lock(lock, alive=lambda p: True)                         # live foreign owner
    assert not ok and holder["cmd"] == "other"
    ok, _ = rg.acquire_lock(lock, alive=lambda p: False)                             # dead owner -> take over
    assert ok
    rg.release_lock(lock)
    assert not lock.exists()


def test_pid_alive_for_current_and_dead_process():
    assert rg.pid_alive(os.getpid())
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert not rg.pid_alive(p.pid)


# ── interpreter choice ────────────────────────────────────────────────────────

def test_python_selection_prefers_3_11_to_3_13_and_rejects_3_14():
    assert rg.py_supported((3, 13)) and rg.py_supported((3, 11)) and not rg.py_supported((3, 14))
    assert not rg.py_supported((3, 10))
    pick = rg.choose_python([("py314", (3, 14)), ("py312", (3, 12)), ("py313", (3, 13)), ("py39", (3, 9))])
    assert pick == ("py313", (3, 13))
    assert rg.choose_python([("py314", (3, 14))]) is None


# ── end-to-end: guarded subprocess ────────────────────────────────────────────

def _run_guard(script_text, tmp_path):
    script = tmp_path / "s.py"
    script.write_text(script_text)
    sentinel = tmp_path / "ok.flag"
    env = {**os.environ, rg.SENTINEL_ENV: str(sentinel), "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-m", "src.utils.run_guard", str(script), "--x", "1"], cwd=ROOT,
                       env=env, capture_output=True, text=True)
    return r, sentinel


def test_guard_writes_sentinel_and_exits_zero_on_success(tmp_path):
    r, sentinel = _run_guard("import sys\nprint('Report saved', sys.argv[1:])\n", tmp_path)
    assert r.returncode == 0 and sentinel.exists() and "Report saved" in r.stdout and "['--x', '1']" in r.stdout


def test_guard_no_sentinel_on_failure(tmp_path):
    r, sentinel = _run_guard("import sys\nsys.exit(3)\n", tmp_path)
    assert r.returncode == 3 and not sentinel.exists()
    r, sentinel = _run_guard("raise RuntimeError('boom')\n", tmp_path)
    assert r.returncode == 1 and not sentinel.exists() and "RuntimeError: boom" in r.stderr
    r, sentinel = _run_guard("import sys\nsys.exit(0)\n", tmp_path)                  # explicit success exit
    assert r.returncode == 0 and sentinel.exists()


def test_finish_called_by_the_script_itself(tmp_path):
    script = tmp_path / "s2.py"
    script.write_text("from src.utils.run_guard import finish\nprint('done')\nfinish()\nprint('never')\n")
    sentinel = tmp_path / "ok2.flag"
    env = {**os.environ, rg.SENTINEL_ENV: str(sentinel), "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and sentinel.exists() and "never" not in r.stdout
