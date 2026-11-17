"""
Run guard: make "the script finished its work" independent of interpreter shutdown.

Problem: on Python 3.14 + native GPU libs (LightGBM OpenCL, XGBoost/CatBoost CUDA, torch)
a script can print "Report saved" and then die with 0xC0000005 while the interpreter tears
the libraries down. A runner that trusts the exit code marks the step FAILED and stops;
~11 GPU-hours were lost that way.

Mechanism
  * The runner (run_all.py) sets FRAUD_STEP_SENTINEL=<file> for every command.
  * A script calls ``finish()`` as its LAST statement (or is started through
    ``python -m src.utils.run_guard script.py args...`` which does it for any script).
    finish() writes the sentinel, flushes stdout/stderr/loguru and ends the process with
    os._exit(code): no atexit handlers, no module teardown, no native destructors.
  * The runner treats "sentinel exists" as success even if the exit code is non-zero (it
    logs a warning), and calls it OUT OF MEMORY only if the log really shows a memory error.

Everything here is stdlib-only so the runner can import it before the venv exists.
Also holds the pure decision helpers the runner uses (unit-tested): ``classify_outcome``,
``looks_like_oom``, ``lock_status``, ``choose_python``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

SENTINEL_ENV = "FRAUD_STEP_SENTINEL"

OOM_MARKERS = (
    "MemoryError", "Unable to allocate", "out of memory", "Out of memory", "CUDA out of memory",
    "std::bad_alloc", "OutOfMemoryError", "cudaErrorMemoryAllocation", "CUDA_ERROR_OUT_OF_MEMORY",
    "bad allocation", "Killed",
)
# exit codes of a process killed by the kernel OOM killer (no Python traceback is possible)
OOM_KILL_CODES = (-9, 137)


# ─────────────────────────────────────────────────────────────────────────────
# script side
# ─────────────────────────────────────────────────────────────────────────────

def _flush_everything() -> None:
    try:
        from loguru import logger
        logger.complete()
        logger.remove()
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def write_sentinel(path: Optional[str] = None) -> Optional[Path]:
    path = path or os.environ.get(SENTINEL_ENV)
    if not path:
        return None
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid()}))
    except OSError:
        return None
    return p


def finish(code: int = 0) -> None:
    """Call as the last statement of a script: sentinel (on success), flush, hard exit."""
    if code == 0:
        write_sentinel()
    _flush_everything()
    os._exit(int(code))


def run_script(script: str, argv: Sequence[str]) -> int:
    """Run ``script`` as __main__ in this process and return its exit code (no shutdown)."""
    import runpy
    import traceback
    script = str(Path(script).resolve())
    sys.argv = [script, *argv]
    sys.path.insert(0, str(Path(script).parent))
    try:
        runpy.run_path(script, run_name="__main__")
        return 0
    except SystemExit as e:
        if e.code in (None, 0):
            return 0
        if isinstance(e.code, int):
            return e.code
        print(e.code, file=sys.stderr)
        return 1
    except BaseException:                                   # noqa: BLE001 - report like Python would
        traceback.print_exc()
        return 1


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        sys.exit("usage: python -m src.utils.run_guard script.py [args...]")
    finish(run_script(argv[0], argv[1:]))


# ─────────────────────────────────────────────────────────────────────────────
# runner side: pure decision helpers
# ─────────────────────────────────────────────────────────────────────────────

def looks_like_oom(rc: int, log_text: str) -> bool:
    """OOM only when the log shows a memory error (or the kernel killed the process)."""
    tail = log_text[-6000:]
    if any(k in tail for k in OOM_MARKERS):
        return True
    return rc in OOM_KILL_CODES and "Traceback (most recent call last)" not in tail


def classify_outcome(rc: int, sentinel_exists: bool, log_text: str = "") -> Tuple[str, Optional[str]]:
    """-> (status, warning).  status in {"done", "oom", "failed"}.

    sentinel present : the script finished its work -> done (warning if rc != 0, e.g. the
                       interpreter crashed during shutdown)
    rc == 0          : done (commands that are not guarded, e.g. pip)
    otherwise        : "oom" if the log shows a memory error, else "failed"
    """
    if sentinel_exists:
        if rc != 0:
            return "done", (f"exit code {rc} (0x{rc & 0xFFFFFFFF:X}) AFTER the script finished its work "
                            f"(sentinel written) - interpreter-shutdown crash of a native library; ignored")
        return "done", None
    if rc == 0:
        return "done", None
    return ("oom" if looks_like_oom(rc, log_text) else "failed"), None


# ── lock file ────────────────────────────────────────────────────────────────

def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(0x1000, False, int(pid))        # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259                              # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_status(info: Optional[Dict], alive: Callable[[int], bool] = pid_alive,
                my_pid: Optional[int] = None) -> str:
    """"free" (no lock) | "stale" (owner dead / unreadable) | "held" (another live run) | "own"."""
    if not info or "pid" not in info:
        return "free" if info is None else "stale"
    pid = int(info["pid"])
    if my_pid is not None and pid == my_pid:
        return "own"
    return "held" if alive(pid) else "stale"


def read_lock(path: Path) -> Optional[Dict]:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return {}                                                 # unreadable -> stale


def acquire_lock(path: Path, extra: Optional[Dict] = None,
                 alive: Callable[[int], bool] = pid_alive) -> Tuple[bool, Optional[Dict]]:
    """Try to take the lock. Returns (acquired, info_of_current_holder_if_not)."""
    path = Path(path)
    info = read_lock(path)
    st = lock_status(info, alive, os.getpid())
    if st == "held":
        return False, info
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                **(extra or {})}))
    return True, None


def release_lock(path: Path) -> None:
    info = read_lock(path)
    if info and info.get("pid") == os.getpid():
        try:
            Path(path).unlink()
        except OSError:
            pass


# ── interpreter choice ───────────────────────────────────────────────────────

SUPPORTED_PY = ((3, 11), (3, 13))          # inclusive range that the pinned wheels are tested on


def py_supported(version: Tuple[int, int]) -> bool:
    return SUPPORTED_PY[0] <= tuple(version[:2]) <= SUPPORTED_PY[1]


def choose_python(candidates: Iterable[Tuple[str, Tuple[int, int]]]) -> Optional[Tuple[str, Tuple[int, int]]]:
    """First supported interpreter, preferring newer minor versions (3.13 > 3.12 > 3.11)."""
    ok = [(exe, v) for exe, v in candidates if py_supported(v)]
    return max(ok, key=lambda c: c[1]) if ok else None


if __name__ == "__main__":
    main()
