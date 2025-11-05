# scheduler_cleanup.py — light-first, collision-safe cleanup scheduler
# (FINAL v1.3 env-aware)
# - predict_lock GC API 정합 유지(정상 락은 절대 건드리지 않음, stale만 정리)
# - start-immediate(부팅 즉시 라이트 1회) 옵션
# - 학습/예측/글로벌락 충돌 회피
# - 디스크 soft/hard cap 기반 라이트/헤비/비상 정리
# - CLI: --start / --once-light / --once-heavy / --stop

import os, sys, time, threading, datetime, pytz, traceback

# 필수 의존: safe_cleanup (프로젝트 내 제공)
import safe_cleanup

# ===== 공통 베이스 경로 (app.py / safe_cleanup.py 와 동일 우선순위) =====
# 1) PERSIST_DIR
# 2) PERSISTENT_DIR
# 3) PERSIST_ROOT
# 4) 없으면 /tmp/persistent
BASE_DIR = (
    os.getenv("PERSIST_DIR")
    or os.getenv("PERSISTENT_DIR")
    or os.getenv("PERSIST_ROOT")
    or "/tmp/persistent"
)

# 선택 의존: train 상태질의 (없으면 항상 False)
try:
    import train
    _is_training = getattr(train, "is_loop_running", lambda: False)
except Exception:
    def _is_training(): return False

# ✅ 선택 의존: predict_lock (stale만 정리)
try:
    import predict_lock as _pl
    _pl_clear_stale = getattr(_pl, "clear_stale_predict_lock", None)  # no-arg, 내부 TTL 사용
except Exception:
    _pl, _pl_clear_stale = None, None

def _clear_stale_predict_lock(tag="cleanup"):
    """예측 락을 강제 삭제하지 않고, predict_lock의 'stale 정리' API만 호출한다."""
    try:
        if _pl_clear_stale is None:
            return
        os.environ["PREDICT_LOCK_GC_TAG"] = str(tag)  # 로깅 힌트
        _pl_clear_stale()
    except Exception:
        pass

# 예측 게이트/락 경로 (app.py/basics와 일치하도록 BASE_DIR 사용)
RUN_DIR        = os.path.join(BASE_DIR, "run")
PREDICT_LOCK   = os.path.join(RUN_DIR, "predict_running.lock")
PREDICT_BLOCK  = os.path.join(BASE_DIR, "predict.block")
PREDICT_GATE   = os.path.join(RUN_DIR, "predict_gate.json")

# 전역 락 (safe_cleanup 이 이미 올바른 LOCK_DIR/LOCK_PATH를 잡아놓음)
LOCK_DIR  = getattr(safe_cleanup, "LOCK_DIR", os.path.join(BASE_DIR, "locks"))
LOCK_PATH = getattr(safe_cleanup, "LOCK_PATH", os.path.join(LOCK_DIR, "train_or_predict.lock"))

# 튜너블 파라미터 (ENV로 오버라이드)
def _env_int(k, d):
    try: return int(os.getenv(k, str(d)))
    except Exception: return d
def _env_float(k, d):
    try: return float(os.getenv(k, str(d)))
    except Exception: return d
def _env_bool(k, d):
    v = os.getenv(k, None)
    return d if v is None else str(v).strip().lower() in {"1","true","y","yes","on"}

CLEAN_INTERVAL_MIN   = _env_int("CLEAN_INTERVAL_MIN", 30)   # 기본 30분(최소 5분)
LIGHT_ONLY_IF_BUSY   = _env_bool("CLEAN_LIGHT_ONLY_IF_BUSY", True)
HEAVY_ALLOW_IF_IDLE  = _env_bool("CLEAN_HEAVY_ALLOW_IF_IDLE", True)
HEAVY_MIN_GAP_MIN    = _env_int("CLEAN_HEAVY_MIN_GAP_MIN", 180)  # heavy 최소 간격 3h
DISK_HARDCAP_GB      = float(getattr(safe_cleanup, "HARD_CAP_GB", 9.6))
DISK_SOFTCAP_GB      = _env_float("CLEAN_SOFTCAP_GB", 8.0)       # soft cap 넘으면 heavy 고려
RUN_ON_START         = _env_bool("CLEAN_RUN_ON_START", True)     # 시작 즉시 light 1회

_tz = pytz.timezone("Asia/Seoul")
_now = lambda: datetime.datetime.now(_tz)

_sched = None
_last_heavy_at = 0.0
_lock = threading.Lock()

def _is_predict_busy() -> bool:
    """예측 타이트 구간 여부(충돌 회피)."""
    try:
        if os.path.exists(PREDICT_LOCK):   # 예측 실행 중
            return True
        if os.path.exists(LOCK_PATH):      # 앱 전역 초기화/정지 중
            return True
        if os.path.exists(PREDICT_BLOCK):  # 강제 차단 중
            return True
        # 게이트가 닫혀 있으면 보수적으로 busy 취급
        if os.path.exists(PREDICT_GATE):
            import json
            with open(PREDICT_GATE, "r", encoding="utf-8") as f:
                if not (json.load(f).get("open", True)):
                    return True
    except Exception:
        return True  # 알 수 없으면 안전하게 busy
    return False

def _last_heavy_ts() -> float:
    global _last_heavy_at
    return float(_last_heavy_at or 0.0)

def _mark_heavy_now():
    global _last_heavy_at
    _last_heavy_at = time.time()

def _should_run_heavy() -> bool:
    """
    무거운 정리를 지금 해도 되는가?
    - 한가(idle)이고
    - 최근 실행 간격 충족 & 디스크 소프트캡 이상
    """
    if not HEAVY_ALLOW_IF_IDLE:
        return False
    if _is_training():
        return False
    if _is_predict_busy():
        return False
    # 최소 간격
    gap_min = (time.time() - _last_heavy_ts()) / 60.0
    if gap_min < HEAVY_MIN_GAP_MIN:
        return False
    # 용량 체크 (BASE_DIR 기준)
    try:
        used = safe_cleanup.get_directory_size_gb(BASE_DIR)
        if used >= DISK_SOFTCAP_GB:
            return True
    except Exception:
        return False
    return False

def _run_light():
    """가벼운 정리: 오래된 로그/캐시/소용량 모델만 제거(빠름)."""
    try:
        safe_cleanup.trigger_light_cleanup()  # 인자 없는 표준 라이트 트리거
        print(f"[CLEANUP] light done @ {_now().strftime('%H:%M:%S')}")
        sys.stdout.flush()
    except Exception as e:
        print(f"[CLEANUP] light error: {e}")
        sys.stdout.flush()

def _run_heavy():
    """무거운 정리: 소프트캡 초과 등 공간 압박 시(한가할 때만)."""
    try:
        _clear_stale_predict_lock(tag="heavy")  # heavy 직전, stale 예측락만 정리
        safe_cleanup.cleanup_logs_and_models()  # 인자 없는 표준 헤비 클리너
        _mark_heavy_now()
        print(f"[CLEANUP] HEAVY done @ {_now().strftime('%H:%M:%S')}")
        sys.stdout.flush()
    except Exception as e:
        print(f"[CLEANUP] heavy error: {e}")
        sys.stdout.flush()

def _cleanup_job():
    """
    스케줄러 틱 본체.
    - busy(학습/예측/전역락/게이트닫힘) → light-only 또는 skip
    - idle & 소프트캡 초과 & 최소간격 충족 → heavy
    - 하드캡 초과 → 비상 정리
    """
    with _lock:
        try:
            used = safe_cleanup.get_directory_size_gb(BASE_DIR)
        except Exception:
            used = -1.0

        # 틱 시작: stale 예측락만 정리(정상 락은 보존)
        _clear_stale_predict_lock(tag="tick")

        used_str = f"{used:.2f}GB" if used >= 0 else "unknown"
        print(f"[CLEANUP] tick busy={_is_training() or _is_predict_busy()} "
              f"used={used_str} (soft={DISK_SOFTCAP_GB:.2f} hard={DISK_HARDCAP_GB:.2f})")
        sys.stdout.flush()

        # 하드캡 초과면 즉시 비상 정리
        try:
            if used >= 0 and used >= DISK_HARDCAP_GB:
                _clear_stale_predict_lock(tag="emergency")
                print("[CLEANUP] EMERGENCY: hard cap exceeded -> run_emergency_purge()")
                sys.stdout.flush()
                safe_cleanup.run_emergency_purge()
                return
        except Exception:
            pass

        busy = _is_training() or _is_predict_busy()
        if busy:
            if LIGHT_ONLY_IF_BUSY:
                _run_light()
            else:
                print("[CLEANUP] busy -> skip")
                sys.stdout.flush()
            return

        # idle
        if _should_run_heavy():
            _run_heavy()
        else:
            _run_light()

def start_cleanup_scheduler():
    """외부 호출: 스케줄러 시작(중복 방지)."""
    global _sched
    if _sched is not None:
        print("[CLEANUP] scheduler already running")
        sys.stdout.flush()
        return

    # apscheduler가 없으면 조용히 포기(의존성 선택)
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except Exception as e:
        print(f"[CLEANUP] apscheduler not available: {e}")
        sys.stdout.flush()
        return

    if RUN_ON_START:
        try:
            _clear_stale_predict_lock(tag="start")
            _run_light()
        except Exception:
            pass

    sched = BackgroundScheduler(
        timezone=_tz,
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 90},
    )
    sched.add_job(
        _cleanup_job,
        trigger="interval",
        minutes=max(5, CLEAN_INTERVAL_MIN),  # 최소 5분
        id="cleanup_tick",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=90,
    )
    sched.start()
    _sched = sched
    print(f"[CLEANUP] scheduler started (every {max(5, CLEAN_INTERVAL_MIN)} min)")
    sys.stdout.flush()

def stop_cleanup_scheduler():
    """외부 호출: 스케줄러 종료."""
    global _sched
    try:
        if _sched is not None:
            try: _sched.pause()
            except Exception: pass
            try: _sched.remove_all_jobs()
            except Exception: pass
            _sched.shutdown(wait=False)
            _sched = None
            print("[CLEANUP] scheduler stopped")
            sys.stdout.flush()
    except Exception as e:
        print(f"[CLEANUP] stop error: {e}")
        sys.stdout.flush()

# ---- 간단 CLI ----
def _cli(argv: list[str]) -> int:
    if not argv:
        print("usage: scheduler_cleanup.py [--start | --once-light | --once-heavy | --stop]")
        return 0
    cmd = argv[0].lower()
    if cmd == "--start":
        start_cleanup_scheduler()
        # 백그라운드 스케줄러를 유지하기 위해 대기 루프
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            stop_cleanup_scheduler()
        return 0
    if cmd == "--once-light":
        _clear_stale_predict_lock(tag="cli-light")
        _run_light()
        return 0
    if cmd == "--once-heavy":
        if _is_training() or _is_predict_busy():
            print("[CLEANUP] cannot run heavy now (busy).")
            return 1
        _run_heavy()
        return 0
    if cmd == "--stop":
        stop_cleanup_scheduler()
        return 0
    print(f"unknown command: {cmd}")
    return 1

if __name__ == "__main__":
    try:
        sys.exit(_cli(sys.argv[1:]))
    except Exception as e:
        print(f"[CLEANUP] main error: {e}")
        traceback.print_exc()
        sys.exit(2)
