# predict_lock.py (FINAL v1.6 — any-lock aware, TTL aligned, safe & backward compatible)
import os, time, atexit, json, glob
from typing import Optional, Tuple, Union, Iterable

# Directories/paths
RUN_DIR   = "/persistent/run"
LOCK_FILE = os.getenv("PREDICT_LOCK_FILE", os.path.join(RUN_DIR, "predict_running.lock"))

# Align default TTL with predict.py
PREDICT_LOCK_TTL = int(os.getenv("PREDICT_LOCK_TTL", "600"))
_WAIT_POLL = float(os.getenv("PREDICT_LOCK_POLL", "0.1"))

__all__ = [
    "LOCK_FILE", "PREDICT_LOCK_TTL",
    "acquire", "release",
    "clear", "clear_stale_predict_lock",
    "wait_clear", "wait_until_free",
    "is_predict_running", "acquire_with_retry",
    "lock_path_for",
]

def _ensure_run_dir():
    try:
        os.makedirs(RUN_DIR, exist_ok=True)
    except Exception:
        pass

def _clean_key(s: str) -> str:
    s = (s or "None")
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)

def lock_path_for(lock_key: Optional[Union[str, Tuple[str, str]]] = None) -> str:
    """None -> legacy lock path; key -> per-pair lock path"""
    _ensure_run_dir()
    if lock_key is None:
        return LOCK_FILE
    if isinstance(lock_key, (tuple, list)) and len(lock_key) >= 2:
        sym, strat = lock_key[0], lock_key[1]
        key = f"{_clean_key(str(sym))}_{_clean_key(str(strat))}"
    else:
        key = _clean_key(str(lock_key))
    return os.path.join(RUN_DIR, f"predict_{key}.lock")

_HELD_LOCKS = set()

def _exists(p: str) -> bool:
    try: return os.path.exists(p)
    except Exception: return False

def _read_lock(p: str):
    try:
        with open(p, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            try: return json.loads(txt)
            except Exception: return {}
    except Exception:
        return {}

def _is_stale(p: str, ttl: Optional[int] = None) -> bool:
    ttl = int(ttl if ttl is not None else PREDICT_LOCK_TTL)
    try:
        if not os.path.exists(p): return False
        mtime = os.path.getmtime(p)
        if time.time() - mtime > max(5, ttl):  # time-based stale
            return True
        info = _read_lock(p)
        pid = int(info.get("pid", -1)) if info else -1
        if pid <= 0: return True
        try:
            # check liveness (POSIX)
            os.kill(pid, 0)
            return False
        except Exception:
            return True
    except Exception:
        # fail-open on errors
        return True

def _clear(p: str, force: bool = False, tag: str = ""):
    try:
        if not _exists(p): return
        os.remove(p)
        print(f"[LOCK] cleared {os.path.basename(p)} ({tag})", flush=True)
    except FileNotFoundError:
        pass
    except Exception as e:
        if force:
            try: os.remove(p)
            except Exception: pass
        print(f"[LOCK] clear error {e} {tag}", flush=True)

def _write_atomic(p: str, content: str) -> bool:
    _ensure_run_dir()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(p, flags)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content); f.flush()
            try: os.fsync(f.fileno())
            except Exception: pass
        return True
    except FileExistsError:
        return False
    except Exception:
        return False

# ---------- helpers to work with *all* locks ----------
def _all_lock_paths() -> Iterable[str]:
    """Return legacy global lock + all per-pair locks."""
    _ensure_run_dir()
    paths = [LOCK_FILE]
    try:
        paths += glob.glob(os.path.join(RUN_DIR, "predict_*.lock"))
    except Exception:
        pass
    # dedup
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def _any_healthy_lock_exists() -> bool:
    any_healthy = False
    for p in _all_lock_paths():
        if not _exists(p): continue
        if _is_stale(p):
            _clear(p, force=True, tag="sweep_stale")
            continue
        any_healthy = True
    return any_healthy

# -------------- Public API --------------
def clear(force: bool = False, stale_sec: Optional[int] = None, tag: str = "", lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    if lock_key is None:
        # clear legacy + stale per-pair
        for p in _all_lock_paths():
            if force or _is_stale(p, stale_sec):
                _clear(p, force=True, tag=(tag or "manual"))
        return
    _clear(lock_path_for(lock_key), force=force, tag=(tag or "manual"))

def wait_clear(timeout_sec: int = 10, stale_sec: Optional[int] = None, poll: float = _WAIT_POLL, tag: str = "", lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    deadline = time.time() + max(1, int(timeout_sec))
    ttl = int(stale_sec) if stale_sec is not None else PREDICT_LOCK_TTL
    if lock_key is None:
        # wait until no healthy locks remain (clear stale asap)
        while time.time() < deadline:
            if not _any_healthy_lock_exists():
                return True
            time.sleep(max(0.05, float(poll)))
        # final sweep
        for p in _all_lock_paths():
            if _is_stale(p, ttl):
                _clear(p, force=True, tag=((tag + "|final").strip("|") or "wait_clear|final"))
        return not _any_healthy_lock_exists()
    # per-key
    p = lock_path_for(lock_key)
    while _exists(p) and time.time() < deadline:
        if _is_stale(p, ttl):
            _clear(p, force=True, tag=(tag or "wait_clear"))
            break
        time.sleep(max(0.05, float(poll)))
    if _exists(p) and _is_stale(p, ttl):
        _clear(p, force=True, tag=((tag + "|final").strip("|") or "wait_clear|final"))
    return not _exists(p)

def acquire(note: str = "", lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    p = lock_path_for(lock_key)
    if _is_stale(p):
        _clear(p, force=True, tag=f"acquire_pre:{note}")
    pid = os.getpid()
    content = json.dumps({"pid": pid, "ts": time.time(), "note": note})
    for attempt in range(0, 10):
        if _write_atomic(p, content):
            _HELD_LOCKS.add(p)
            print(f"[LOCK] acquired {os.path.basename(p)} pid={pid} note={note}", flush=True)
            return True
        if _is_stale(p):
            _clear(p, force=True, tag=f"acquire_retry:{note}")
            time.sleep(0.05); continue
        time.sleep(0.05 + 0.05 * attempt)
    raise RuntimeError(f"Failed to acquire lock {p} after retries")

def release(note: str = "", lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    p = lock_path_for(lock_key)
    try:
        if not _exists(p): return True
        info = _read_lock(p)
        pid = int(info.get("pid", -1)) if info else -1
        if pid == os.getpid() or _is_stale(p, 0):
            _clear(p, force=True, tag=f"release:{note}")
            _HELD_LOCKS.discard(p)
            print(f"[LOCK] released {os.path.basename(p)} note={note}", flush=True)
            return True
        print(f"[LOCK] release skipped (not owner) file={os.path.basename(p)} note={note}", flush=True)
        return False
    except Exception as e:
        print(f"[LOCK] release exception: {e} note={note}", flush=True)
        _clear(p, force=True, tag=f"release_force:{note}")
        return True

def is_predict_running(lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    """
    None  -> busy if ANY healthy lock exists (legacy or per-pair).
    key   -> busy if that lock exists and is healthy.
    Stale locks are cleared on sight.
    """
    if lock_key is None:
        return _any_healthy_lock_exists()
    p = lock_path_for(lock_key)
    if not _exists(p): return False
    if _is_stale(p):
        _clear(p, force=True, tag="is_running_stale")
        return False
    return True

def clear_stale_predict_lock(lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    if lock_key is None:
        for p in _all_lock_paths():
            if _is_stale(p):
                _clear(p, force=True, tag="clear_stale_api")
        return
    p = lock_path_for(lock_key)
    if _is_stale(p):
        _clear(p, force=True, tag="clear_stale_api")

def wait_until_free(max_wait_sec: int, lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    """
    None -> wait until ALL locks are free (clears stale immediately).
    key  -> wait until that single lock is free.
    """
    start = time.time()
    if lock_key is None:
        while time.time() - start < max(1, int(max_wait_sec)):
            if not _any_healthy_lock_exists():
                return True
            time.sleep(_WAIT_POLL)
        return not _any_healthy_lock_exists()
    # per-key
    p = lock_path_for(lock_key)
    if not _exists(p): return True
    if _is_stale(p):
        _clear(p, force=True, tag="wait_until_free"); return True
    while _exists(p) and (time.time() - start) < max(1, int(max_wait_sec)):
        if _is_stale(p):
            _clear(p, force=True, tag="wait_until_free"); return True
        time.sleep(_WAIT_POLL)
    return not _exists(p)

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

def _atexit_clear():
    try:
        for p in list(_HELD_LOCKS):
            if _exists(p):
                _clear(p, force=True, tag="atexit")
    except Exception:
        pass

atexit.register(_atexit_clear)
