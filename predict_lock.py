# predict_lock.py (FINAL: per-key atomic lock, owner-check, wait/clear API)
import os
import time
import atexit
import json
from typing import Optional, Tuple, Union

# ── 기본 전역(하위호환용)
LOCK_FILE = os.getenv("PREDICT_LOCK_FILE", "/persistent/run/predict_running.lock")

# 기본 TTL / 폴링
_DEFAULT_STALE = int(os.getenv("PREDICT_LOCK_STALE_SEC", "60"))
PREDICT_LOCK_TTL = int(os.getenv("PREDICT_LOCK_TTL", str(_DEFAULT_STALE)))
_WAIT_POLL = float(os.getenv("PREDICT_LOCK_POLL", "0.1"))

__all__ = [
    "LOCK_FILE", "PREDICT_LOCK_TTL",
    "acquire", "release",
    "clear", "clear_stale_predict_lock",
    "wait_clear", "wait_until_free",
    "is_predict_running", "acquire_with_retry",
    "lock_path_for",
]

# ──────────────────────────────────────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────────────────────────────────────

def _clean_key(s: str) -> str:
    s = (s or "None")
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)

def lock_path_for(lock_key: Optional[Union[str, Tuple[str, str]]] = None) -> str:
    """
    lock_key:
      - None  -> legacy 단일 락 (LOCK_FILE)
      - "BTCUSDT_단기" 또는 ("BTCUSDT","단기")
    """
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

# atexit 정리를 위해 내가 보유한 락 경로를 기억
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
                parts = dict(item.split("=", 1) for item in txt.split() if "=" in item)
                return {"pid": int(parts.get("pid", -1)), "ts": float(parts.get("ts", 0)), "note": parts.get("note", "")}
    except Exception:
        return None

def _is_stale(path: str, stale_sec: Optional[int] = None) -> bool:
    try:
        if not os.path.exists(path):
            return False
        ttl = PREDICT_LOCK_TTL if stale_sec is None else int(stale_sec)
        mtime = os.path.getmtime(path)
        age = time.time() - mtime
        if age > max(5, ttl):
            return True
        info = _read_lock(path)
        if info and "pid" in info:
            pid = int(info.get("pid", -1))
            if pid <= 0:
                return True
            try:
                # pid 살아있는지 체크 (권한 불필요한 0 시그널)
                os.kill(pid, 0)
                return False
            except Exception:
                return True
        return False
    except Exception:
        return False

def _clear(path: str, force: bool = False, stale_sec: Optional[int] = None, tag: str = ""):
    try:
        if not _exists(path):
            return
        ttl = PREDICT_LOCK_TTL if stale_sec is None else int(stale_sec)
        if force or _is_stale(path, ttl):
            try:
                os.remove(path)
                print(f"[LOCK] cleared {os.path.basename(path)} (force={int(bool(force))}) {tag}", flush=True)
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[LOCK] clear fail remove: {e} {tag}", flush=True)
    except Exception as e:
        print(f"[LOCK] clear fail: {e} {tag}", flush=True)

def _write_atomic(path: str, content: str) -> bool:
    """O_CREAT|O_EXCL 원자적 생성."""
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
    except Exception as e:
        # best-effort fallback
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            return True
        except Exception:
            print(f"[LOCK] atomic write fail: {e}", flush=True)
            return False

# ──────────────────────────────────────────────────────────────────────────────
# 공개 API (모두 lock_key 인자 지원, 미지정 시 legacy 파일 사용)
# ──────────────────────────────────────────────────────────────────────────────

def clear(force: bool = False, stale_sec: Optional[int] = None, tag: str = "", lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    _clear(lock_path_for(lock_key), force=force, stale_sec=stale_sec, tag=tag)

def wait_clear(timeout_sec: int = 10, stale_sec: Optional[int] = None, poll: float = _WAIT_POLL, tag: str = "", lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    path = lock_path_for(lock_key)
    deadline = time.time() + max(1, int(timeout_sec))
    ttl = PREDICT_LOCK_TTL if stale_sec is None else int(stale_sec)
    while _exists(path) and time.time() < deadline:
        try:
            if _is_stale(path, ttl):
                _clear(path, force=True, stale_sec=ttl, tag=tag or "wait_clear")
                break
        except Exception:
            pass
        time.sleep(max(0.05, float(poll)))
    # 마지막 안전장치
    if _exists(path):
        _clear(path, force=True, stale_sec=ttl, tag=(tag + "|final").strip("|"))
    return not _exists(path)

def acquire(note: str = "", lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    """
    락 획득(원자적). stale 기준은 PREDICT_LOCK_TTL 사용.
    lock_key로 심볼/전략별 락을 생성한다.
    """
    path = lock_path_for(lock_key)
    # stale만 정리(정상 보유는 건드리지 않음)
    _clear(path, force=False, stale_sec=PREDICT_LOCK_TTL, tag=f"acquire_pre:{note}")
    pid = os.getpid()
    content = json.dumps({"pid": pid, "ts": time.time(), "note": note})
    for attempt in range(0, 6):
        ok = _write_atomic(path, content)
        if ok:
            _HELD_LOCKS.add(path)
            print(f"[LOCK] acquired {os.path.basename(path)} pid={pid} note={note}", flush=True)
            return True
        # 이미 있고 stale 아님 → 짧게 백오프 후 재시도
        if _exists(path) and not _is_stale(path):
            time.sleep(0.05 + attempt * 0.05)
            continue
        # stale이면 정리 후 즉시 재시도
        if _is_stale(path):
            _clear(path, force=True, stale_sec=PREDICT_LOCK_TTL, tag=f"acquire_retry:{note}")
            time.sleep(0.05)
            continue
    raise RuntimeError(f"Failed to acquire lock {path} after retries")

def release(note: str = "", lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    """소유자만 해제(또는 stale이면 안전 해제)."""
    path = lock_path_for(lock_key)
    try:
        info = _read_lock(path)
        if info and "pid" in info and int(info.get("pid", -1)) == os.getpid():
            try:
                os.remove(path)
                _HELD_LOCKS.discard(path)
                print(f"[LOCK] released by owner pid={os.getpid()} file={os.path.basename(path)} note={note}", flush=True)
                return True
            except FileNotFoundError:
                _HELD_LOCKS.discard(path)
                return True
            except Exception as e:
                print(f"[LOCK] release fail: {e} {note}", flush=True)
                return False
        else:
            if _is_stale(path, stale_sec=0):
                try:
                    os.remove(path)
                    print(f"[LOCK] released stale by non-owner pid={os.getpid()} file={os.path.basename(path)} note={note}", flush=True)
                    return True
                except Exception as e:
                    print(f"[LOCK] non-owner clear fail: {e} {note}", flush=True)
                    return False
            print(f"[LOCK] release skipped (not owner) file={os.path.basename(path)} note={note}", flush=True)
            return False
    except Exception as e:
        print(f"[LOCK] release exception: {e} {note}", flush=True)
        return False

def is_predict_running(lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    """예측 락이 살아있는지. stale이면 즉시 정리 후 False."""
    path = lock_path_for(lock_key)
    if not _exists(path):
        return False
    if _is_stale(path):
        _clear(path, force=True, stale_sec=PREDICT_LOCK_TTL, tag="is_running_stale")
        return False
    return True

def clear_stale_predict_lock(lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    """TTL 기준으로만 stale 정리."""
    _clear(lock_path_for(lock_key), force=False, stale_sec=PREDICT_LOCK_TTL, tag="clear_stale_api")

def wait_until_free(max_wait_sec: int, lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    """게이트가 풀릴 때까지 대기. (stale는 TTL 기준 자동정리)"""
    return wait_clear(timeout_sec=max_wait_sec, stale_sec=PREDICT_LOCK_TTL, poll=_WAIT_POLL, tag="wait_until_free", lock_key=lock_key)

def acquire_with_retry(max_wait_sec: int, note: str = "", lock_key: Optional[Union[str, Tuple[str, str]]] = None):
    """deadline 내에서 락 획득 재시도."""
    deadline = time.time() + max(1, int(max_wait_sec))
    while time.time() < deadline:
        try:
            if acquire(note=note, lock_key=lock_key):
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False

# ───── 프로세스 종료 시 내가 가진 모든 락 정리 ─────
def _atexit_clear():
    try:
        for path in list(_HELD_LOCKS):
            info = _read_lock(path)
            if info and "pid" in info and int(info.get("pid", -1)) == os.getpid():
                _clear(path, force=True, stale_sec=0, tag="atexit")
    except Exception:
        pass

atexit.register(_atexit_clear)
