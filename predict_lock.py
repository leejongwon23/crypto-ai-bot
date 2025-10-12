# predict_lock.py (FINAL v1.5 — per-key atomic lock, auto-clear on stale, safe-release)
import os
import time
import atexit
import json
from typing import Optional, Tuple, Union

# Legacy single lock path (kept for backward compatibility)
LOCK_FILE = os.getenv("PREDICT_LOCK_FILE", "/persistent/run/predict_running.lock")

# Short TTL so stale locks are cleared quickly
PREDICT_LOCK_TTL = int(os.getenv("PREDICT_LOCK_TTL", "30"))
_WAIT_POLL = float(os.getenv("PREDICT_LOCK_POLL", "0.1"))

__all__ = [
    "LOCK_FILE", "PREDICT_LOCK_TTL",
    "acquire", "release",
    "clear", "clear_stale_predict_lock",
    "wait_clear", "wait_until_free",
    "is_predict_running", "acquire_with_retry",
    "lock_path_for",
]

# ───────────────────────────────────────────────
# Internals
# ───────────────────────────────────────────────
def _clean_key(s: str) -> str:
    s = (s or "None")
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)

def lock_path_for(lock_key: Optional[Union[str, Tuple[str, str]]] = None) -> str:
    """lock_key: None -> legacy single lock; 'BTCUSDT_단기' or ('BTCUSDT','단기') -> per-key lock"""
    if lock_key is None:
        return LOCK_FILE
    if isinstance(lock_key, (tuple, list)) and len(lock_key) >= 2:
        sym, strat = lock_key[0], lock_key[1]
        key = f"{_clean_key(str(sym))}_{_clean_key(str(strat))}"
    else:
        key = _clean_key(str(lock_key))
    run_dir = "/persistent/run"
    try:
        os.makedirs(run_dir, exist_ok=True)
    except Exception:
        pass
    return os.path.join(run_dir, f"predict_{key}.lock")

_HELD_LOCKS = set()

def _exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False

def _read_lock(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            try:
                return json.loads(txt)
            except Exception:
                return {}
    except Exception:
        return {}

def _is_stale(path: str, ttl: Optional[int] = None) -> bool:
    ttl = ttl or PREDICT_LOCK_TTL
    try:
        if not os.path.exists(path):
            return False
        mtime = os.path.getmtime(path)
        # time-based staleness
        if time.time() - mtime > max(5, ttl):
            return True
        # owner process liveness
        info = _read_lock(path)
        pid = int(info.get("pid", -1)) if info else -1
        if pid <= 0:
            return True
        try:
            os.kill(pid, 0)  # does not actually kill; checks existence
            return False
        except Exception:
            return True
    except Exception:
        # On any error, treat as stale to fail open
        return True

def _clear(path: str, force: bool = False, tag: str = ""):
    try:
        if not _exists(path):
            return
        os.remove(path)
        print(f"[LOCK] cleared {os.path.basename(path)} ({tag})", flush=True)
    except FileNotFoundError:
        pass
    except Exception as e:
        if force:
            try:
                os.remove(path)
            except Exception:
                pass
        print(f"[LOCK] clear error {e} {tag}", flush=True)

def _write_atomic(path: str, content: str) -> bool:
    """Create the lock file atomically (O_CREAT|O_EXCL)."""
    dirp = os.path.dirname(path)
    if dirp:
        try:
            os.makedirs(dirp, exist_ok=True)
        except Exception:
            pass
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        return True
    except FileExistsError:
        return False
    except Exception:
        return False

# ───────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────
def clear(force: bool = False, stale_sec: Optional[int] = None, tag: str = "", lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    _clear(lock_path_for(lock_key), force=force, tag=(tag or "manual"))

def wait_clear(timeout_sec: int = 10, stale_sec: Optional[int] = None, poll: float = _WAIT_POLL, tag: str = "", lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    path = lock_path_for(lock_key)
    deadline = time.time() + max(1, int(timeout_sec))
    ttl = int(stale_sec) if stale_sec is not None else PREDICT_LOCK_TTL
    while _exists(path) and time.time() < deadline:
        if _is_stale(path, ttl):
            _clear(path, force=True, tag=(tag or "wait_clear"))
            break
        time.sleep(max(0.05, float(poll)))
    # final safeguard
    if _exists(path) and _is_stale(path, ttl):
        _clear(path, force=True, tag=((tag + "|final").strip("|") or "wait_clear|final"))
    return not _exists(path)

def acquire(note: str = "", lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    """
    Acquire per-key lock atomically. Stale locks are cleared automatically.
    """
    path = lock_path_for(lock_key)
    # Pre-clear stale only (don't touch healthy locks)
    if _is_stale(path):
        _clear(path, force=True, tag=f"acquire_pre:{note}")
    pid = os.getpid()
    content = json.dumps({"pid": pid, "ts": time.time(), "note": note})
    for attempt in range(0, 10):
        if _write_atomic(path, content):
            _HELD_LOCKS.add(path)
            print(f"[LOCK] acquired {os.path.basename(path)} pid={pid} note={note}", flush=True)
            return True
        # If someone holds it but it turns stale, clear and retry
        if _is_stale(path):
            _clear(path, force=True, tag=f"acquire_retry:{note}")
            time.sleep(0.05)
            continue
        time.sleep(0.05 + 0.05 * attempt)
    raise RuntimeError(f"Failed to acquire lock {path} after retries")

def release(note: str = "", lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    """
    Release only if owner OR if the lock is stale (to guarantee no orphan locks left).
    """
    path = lock_path_for(lock_key)
    try:
        if not _exists(path):
            return True
        info = _read_lock(path)
        pid = int(info.get("pid", -1)) if info else -1
        if pid == os.getpid() or _is_stale(path, 0):
            _clear(path, force=True, tag=f"release:{note}")
            _HELD_LOCKS.discard(path)
            print(f"[LOCK] released {os.path.basename(path)} note={note}", flush=True)
            return True
        print(f"[LOCK] release skipped (not owner) file={os.path.basename(path)} note={note}", flush=True)
        return False
    except Exception as e:
        print(f"[LOCK] release exception: {e} note={note}", flush=True)
        _clear(path, force=True, tag=f"release_force:{note}")
        return True

def is_predict_running(lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    """Return False if lock is stale (and clear it), True if healthy."""
    path = lock_path_for(lock_key)
    if not _exists(path):
        return False
    if _is_stale(path):
        _clear(path, force=True, tag="is_running_stale")
        return False
    return True

def clear_stale_predict_lock(lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    """Clear only when considered stale."""
    path = lock_path_for(lock_key)
    if _is_stale(path):
        _clear(path, force=True, tag="clear_stale_api")

def wait_until_free(max_wait_sec: int, lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    """
    Wait for a healthy lock to free. If stale is detected at any point, clear immediately and return True.
    """
    path = lock_path_for(lock_key)
    if not _exists(path):
        return True
    if _is_stale(path):
        _clear(path, force=True, tag="wait_until_free")
        return True
    start = time.time()
    while _exists(path) and (time.time() - start) < max(1, int(max_wait_sec)):
        if _is_stale(path):
            _clear(path, force=True, tag="wait_until_free")
            return True
        time.sleep(_WAIT_POLL)
    return not _exists(path)

def acquire_with_retry(max_wait_sec: int, note: str = "", lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    deadline = time.time() + max(1, int(max_wait_sec))
    while time.time() < deadline:
        try:
            if acquire(note=note, lock_key=lock_key):
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False

# Ensure all held locks are cleared on process exit
def _atexit_clear():
    try:
        for path in list(_HELD_LOCKS):
            if _exists(path):
                _clear(path, force=True, tag="atexit")
    except Exception:
        pass

atexit.register(_atexit_clear)
