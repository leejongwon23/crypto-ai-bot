# predict_lock.py (PATCHED: atomic create + owner-check + robust wait/clear)
import os
import time
import atexit
import json

LOCK_FILE = os.getenv("PREDICT_LOCK_FILE", "/persistent/run/predict_running.lock")
# 최소 간격/기본 stale 초
_DEFAULT_STALE = int(os.getenv("PREDICT_LOCK_STALE_SEC", "60"))
_WAIT_POLL = float(os.getenv("PREDICT_LOCK_POLL", "0.1"))

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
                # legacy plain text
                parts = dict(item.split("=", 1) for item in txt.split() if "=" in item)
                return {"pid": int(parts.get("pid", -1)), "ts": float(parts.get("ts", 0)), "note": parts.get("note", "")}
    except Exception:
        return None

def _is_stale(stale_sec=_DEFAULT_STALE):
    try:
        if not os.path.exists(LOCK_FILE):
            return False
        info = _read_lock()
        mtime = os.path.getmtime(LOCK_FILE)
        age = time.time() - mtime
        if age > max(5, int(stale_sec)):
            return True
        # 추가 검사: pid가 없거나 프로세스가 없음
        if info and "pid" in info:
            pid = int(info.get("pid", -1))
            if pid <= 0:
                return True
            try:
                # signal 0 check
                os.kill(pid, 0)
                return False
            except Exception:
                return True
        return False
    except Exception:
        return False

def clear(force=False, stale_sec=_DEFAULT_STALE, tag=""):
    try:
        if not _exists():
            return
        if force or _is_stale(stale_sec):
            try:
                os.remove(LOCK_FILE)
                print(f"[LOCK] cleared {LOCK_FILE} (force={int(bool(force))}) {tag}", flush=True)
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[LOCK] clear fail remove: {e} {tag}", flush=True)
    except Exception as e:
        print(f"[LOCK] clear fail: {e} {tag}", flush=True)

def wait_clear(timeout_sec=10, stale_sec=_DEFAULT_STALE, poll=_WAIT_POLL, tag=""):
    deadline = time.time() + max(1, int(timeout_sec))
    while _exists() and time.time() < deadline:
        try:
            if _is_stale(stale_sec):
                clear(force=True, stale_sec=stale_sec, tag=tag or "wait_clear")
                break
        except Exception:
            pass
        time.sleep(max(0.05, float(poll)))
    # 마지막 안전장치: 강제 삭제 시도
    if _exists():
        clear(force=True, stale_sec=stale_sec, tag=(tag + "|final").strip("|"))
    return not _exists()

def _write_atomic(content: str):
    """
    안전한 원자적 파일 생성: O_CREAT|O_EXCL 사용.
    경우에 따라 파일이 이미 존재하면 FileExistsError 발생.
    """
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
        # fallback: try normal write (best-effort)
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
    """
    Acquire lock atomically. If another process holds it, wait a short period.
    Raises RuntimeError if cannot acquire within retries.
    """
    # attempt quick clear of stale
    wait_clear(timeout_sec=2, stale_sec=30, poll=0.05, tag=f"acquire:{note}")
    pid = os.getpid()
    content = json.dumps({"pid": pid, "ts": time.time(), "note": note})
    # try a few times to create atomically
    for attempt in range(0, 6):
        ok = _write_atomic(content)
        if ok:
            # success
            print(f"[LOCK] acquired {LOCK_FILE} pid={pid} note={note}", flush=True)
            return True
        # if exists and not stale, short backoff and retry
        if _exists() and not _is_stale():
            time.sleep(0.05 + attempt * 0.05)
            continue
        # try clearing stale and retry immediately
        if _is_stale():
            clear(force=True, stale_sec=0, tag=f"acquire_retry:{note}")
            time.sleep(0.05)
            continue
    raise RuntimeError(f"Failed to acquire lock {LOCK_FILE} after retries")

def release(note=""):
    """
    Release lock only if owned or force removal requested via clear(force=True).
    """
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
            # not owner: attempt safe clear only if stale
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

# ensure lock file is cleared on process exit if owned
def _atexit_clear():
    try:
        info = _read_lock()
        if info and "pid" in info and int(info.get("pid", -1)) == os.getpid():
            clear(force=True, stale_sec=0, tag="atexit")
    except Exception:
        pass

atexit.register(_atexit_clear)
