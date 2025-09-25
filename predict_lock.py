# predict_lock.py (SOT FINAL: atomic create + owner-check + wait/clear API)
import os
import time
import atexit
import json

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
]

def _exists():
    try:
        return os.path.exists(LOCK_FILE)
    except Exception:
        return False

def _read_lock():
    try:
        with open(LOCK_FILE, "r", encoding="utf-8") as f:
            txt = f.read()
            try:
                return json.loads(txt)
            except Exception:
                parts = dict(item.split("=", 1) for item in txt.split() if "=" in item)
                return {"pid": int(parts.get("pid", -1)), "ts": float(parts.get("ts", 0)), "note": parts.get("note", "")}
    except Exception:
        return None

def _is_stale(stale_sec=None):
    try:
        if not os.path.exists(LOCK_FILE):
            return False
        ttl = PREDICT_LOCK_TTL if stale_sec is None else int(stale_sec)
        mtime = os.path.getmtime(LOCK_FILE)
        age = time.time() - mtime
        if age > max(5, ttl):
            return True
        info = _read_lock()
        if info and "pid" in info:
            pid = int(info.get("pid", -1))
            if pid <= 0:
                return True
            try:
                os.kill(pid, 0)  # 존재 확인
                return False
            except Exception:
                return True
        return False
    except Exception:
        return False

def clear(force=False, stale_sec=None, tag=""):
    try:
        if not _exists():
            return
        ttl = PREDICT_LOCK_TTL if stale_sec is None else int(stale_sec)
        if force or _is_stale(ttl):
            try:
                os.remove(LOCK_FILE)
                print(f"[LOCK] cleared {LOCK_FILE} (force={int(bool(force))}) {tag}", flush=True)
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[LOCK] clear fail remove: {e} {tag}", flush=True)
    except Exception as e:
        print(f"[LOCK] clear fail: {e} {tag}", flush=True)

def wait_clear(timeout_sec=10, stale_sec=None, poll=_WAIT_POLL, tag=""):
    deadline = time.time() + max(1, int(timeout_sec))
    ttl = PREDICT_LOCK_TTL if stale_sec is None else int(stale_sec)
    while _exists() and time.time() < deadline:
        try:
            if _is_stale(ttl):
                clear(force=True, stale_sec=ttl, tag=tag or "wait_clear")
                break
        except Exception:
            pass
        time.sleep(max(0.05, float(poll)))
    # 마지막 안전장치
    if _exists():
        clear(force=True, stale_sec=ttl, tag=(tag + "|final").strip("|"))
    return not _exists()

def _write_atomic(content: str):
    """O_CREAT|O_EXCL 원자적 생성."""
    dirp = os.path.dirname(LOCK_FILE)
    if dirp:
        os.makedirs(dirp, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(LOCK_FILE, flags)
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
            with open(LOCK_FILE, "w", encoding="utf-8") as f:
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

def acquire(note=""):
    """락 획득(원자적). stale 기준은 PREDICT_LOCK_TTL 사용."""
    # stale만 정리(정상 보유는 건드리지 않음)
    clear(force=False, stale_sec=PREDICT_LOCK_TTL, tag=f"acquire_pre:{note}")
    pid = os.getpid()
    content = json.dumps({"pid": pid, "ts": time.time(), "note": note})
    for attempt in range(0, 6):
        ok = _write_atomic(content)
        if ok:
            print(f"[LOCK] acquired {LOCK_FILE} pid={pid} note={note}", flush=True)
            return True
        # 이미 있고 stale 아님 → 짧게 백오프 후 재시도
        if _exists() and not _is_stale():
            time.sleep(0.05 + attempt * 0.05)
            continue
        # stale이면 정리 후 즉시 재시도
        if _is_stale():
            clear(force=True, stale_sec=PREDICT_LOCK_TTL, tag=f"acquire_retry:{note}")
            time.sleep(0.05)
            continue
    raise RuntimeError(f"Failed to acquire lock {LOCK_FILE} after retries")

def release(note=""):
    """소유자만 해제(또는 stale이면 안전 해제)."""
    try:
        info = _read_lock()
        if info and "pid" in info and int(info.get("pid", -1)) == os.getpid():
            try:
                os.remove(LOCK_FILE)
                print(f"[LOCK] released by owner pid={os.getpid()} note={note}", flush=True)
                return True
            except FileNotFoundError:
                return True
            except Exception as e:
                print(f"[LOCK] release fail: {e} {note}", flush=True)
                return False
        else:
            if _is_stale(stale_sec=0):
                try:
                    os.remove(LOCK_FILE)
                    print(f"[LOCK] released stale by non-owner pid={os.getpid()} note={note}", flush=True)
                    return True
                except Exception as e:
                    print(f"[LOCK] non-owner clear fail: {e} {note}", flush=True)
                    return False
            print(f"[LOCK] release skipped (not owner) note={note}", flush=True)
            return False
    except Exception as e:
        print(f"[LOCK] release exception: {e} {note}", flush=True)
        return False

# ───── 중앙 API (predict_trigger/predict에서 선택적으로 사용) ─────
def is_predict_running():
    """예측 락이 살아있는지. stale이면 즉시 정리 후 False."""
    if not _exists():
        return False
    if _is_stale():
        clear(force=True, stale_sec=PREDICT_LOCK_TTL, tag="is_running_stale")
        return False
    return True

def clear_stale_predict_lock():
    """TTL 기준으로만 stale 정리."""
    clear(force=False, stale_sec=PREDICT_LOCK_TTL, tag="clear_stale_api")

def wait_until_free(max_wait_sec: int):
    """게이트가 풀릴 때까지 대기. (stale는 TTL 기준 자동정리)"""
    return wait_clear(timeout_sec=max_wait_sec, stale_sec=PREDICT_LOCK_TTL, poll=_WAIT_POLL, tag="wait_until_free")

def acquire_with_retry(max_wait_sec: int, note=""):
    """deadline 내에서 락 획득 재시도."""
    deadline = time.time() + max(1, int(max_wait_sec))
    while time.time() < deadline:
        try:
            if acquire(note=note):
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False

# 프로세스 종료 시 소유 중 락 정리
def _atexit_clear():
    try:
        info = _read_lock()
        if info and "pid" in info and int(info.get("pid", -1)) == os.getpid():
            clear(force=True, stale_sec=0, tag="atexit")
    except Exception:
        pass

atexit.register(_atexit_clear)
