# predict_lock.py
import os, time, atexit

LOCK_FILE = os.getenv("PREDICT_LOCK_FILE", "/persistent/run/predict_running.lock")

def _exists():
    try: return os.path.exists(LOCK_FILE)
    except: return False

def _is_stale(stale_sec=60):
    try:
        if not os.path.exists(LOCK_FILE): return False
        return (time.time() - os.path.getmtime(LOCK_FILE)) > max(5, int(stale_sec))
    except:
        return False

def clear(force=False, stale_sec=60, tag=""):
    try:
        if not _exists(): return
        if force or _is_stale(stale_sec):
            os.remove(LOCK_FILE)
            print(f"[LOCK] cleared {LOCK_FILE} (force={int(force)}) {tag}", flush=True)
    except Exception as e:
        print(f"[LOCK] clear fail: {e} {tag}", flush=True)

def wait_clear(timeout_sec=10, stale_sec=60, poll=0.1, tag=""):
    deadline = time.time() + max(1, int(timeout_sec))
    while _exists() and time.time() < deadline:
        if _is_stale(stale_sec):
            clear(force=True, stale_sec=stale_sec, tag=tag or "wait_clear")
            break
        time.sleep(max(0.05, float(poll)))
    if _exists():  # 마지막 안전장치
        clear(force=True, stale_sec=stale_sec, tag=(tag + "|final").strip("|"))
    return not _exists()

def acquire(note=""):
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    wait_clear(timeout_sec=2, stale_sec=30, poll=0.1, tag=f"acquire:{note}")
    with open(LOCK_FILE, "w") as f:
        f.write(f"pid={os.getpid()} ts={time.time()} note={note}")
        f.flush(); os.fsync(f.fileno())

def release(note=""):
    clear(force=True, stale_sec=0, tag=f"release:{note}")

# 프로세스가 죽을 때도 문을 꼭 연다
atexit.register(lambda: clear(force=True, stale_sec=0, tag="atexit"))
