# scheduler_cleanup.py — light-first, collision-safe cleanup scheduler
# - 학습/예측 타이트 구간에서는 '가벼운 모드'만 실행하거나 스킵
# - 무거운 정리는 한가할 때만 (학습/락/게이트 닫힘/예측락 등 없을 때)
# - app.py 에서 start_cleanup_scheduler(), stop_cleanup_scheduler() 사용

import os, sys, time, threading, datetime, pytz, traceback
from apscheduler.schedulers.background import BackgroundScheduler

# 필수 의존: safe_cleanup(이미 프로젝트에 존재)
import safe_cleanup

# 선택 의존: train 모듈의 상태질의 (없으면 항상 False)
try:
    import train
    _is_training = getattr(train, "is_loop_running", lambda: False)
except Exception:
    def _is_training(): return False

# 예측 게이트/락 경로 (predict.py와 합)
RUN_DIR        = "/persistent/run"
PREDICT_LOCK   = os.path.join(RUN_DIR, "predict_running.lock")
PREDICT_BLOCK  = "/persistent/predict.block"
PREDICT_GATE   = os.path.join(RUN_DIR, "predict_gate.json")

# 전역 락 (app.py/safe_cleanup 에서 공유)
LOCK_DIR  = getattr(safe_cleanup, "LOCK_DIR", "/persistent/locks")
LOCK_PATH = getattr(safe_cleanup, "LOCK_PATH", os.path.join(LOCK_DIR, "train_or_predict.lock"))

# 튜너블 파라미터 (환경변수로 오버라이드 가능)
CLEAN_INTERVAL_MIN   = int(os.getenv("CLEAN_INTERVAL_MIN", "30"))  # 기본 30분
LIGHT_ONLY_IF_BUSY   = os.getenv("CLEAN_LIGHT_ONLY_IF_BUSY", "1") == "1"
HEAVY_ALLOW_IF_IDLE  = os.getenv("CLEAN_HEAVY_ALLOW_IF_IDLE", "1") == "1"
HEAVY_MIN_GAP_MIN    = int(os.getenv("CLEAN_HEAVY_MIN_GAP_MIN", "180"))  # 무거운 정리 최소 간격(3h)
DISK_HARDCAP_GB      = float(getattr(safe_cleanup, "HARD_CAP_GB", 9.6))
DISK_SOFTCAP_GB      = float(os.getenv("CLEAN_SOFTCAP_GB", "8.0"))      # 소프트 캡(넘으면 heavy 고려)

_tz = pytz.timezone("Asia/Seoul")
_now = lambda: datetime.datetime.now(_tz)

_sched = None
_last_heavy_at = 0.0
_lock = threading.Lock()

def _is_predict_busy() -> bool:
    """예측 타이트 구간 여부."""
    try:
        if os.path.exists(PREDICT_LOCK):   # 예측 실행 중 표시
            return True
        if os.path.exists(LOCK_PATH):      # 앱 전역 초기화/정지 중
            return True
        if os.path.exists(PREDICT_BLOCK):  # 강제 차단 중이면 충돌 가능성 -> busy 취급
            return True
        # 게이트가 닫혀 있으면 학습/초기화 등과 겹칠 확률 높다고 보고 보수적으로 busy
        if os.path.exists(PREDICT_GATE):
            import json
            with open(PREDICT_GATE, "r", encoding="utf-8") as f:
                if not json.load(f).get("open", True):
                    return True
    except Exception:
        # 알 수 없으면 안전하게 busy 취급
        return True
    return False

def _should_run_heavy() -> bool:
    """무거운 정리를 지금 해도 되는가? (완전 한가 + 최소 간격 + 용량 압박)"""
    if not HEAVY_ALLOW_IF_IDLE:
        return False
    if _is_training():
        return False
    if _is_predict_busy():
        return False
    # 최근 heavy 이후 최소 간격
    gap_min = (time.time() - _last_heavy_ts()) / 60.0
    if gap_min < HEAVY_MIN_GAP_MIN:
        return False
    # 용량 상태 확인
    try:
        used = safe_cleanup.get_directory_size_gb("/persistent")
        if used >= DISK_SOFTCAP_GB:
            return True
    except Exception:
        # 상태를 못 읽으면 heavy는 미루자
        return False
    return False

def _last_heavy_ts() -> float:
    global _last_heavy_at
    return float(_last_heavy_at or 0.0)

def _mark_heavy_now():
    global _last_heavy_at
    _last_heavy_at = time.time()

def _run_light():
    """가벼운 모드: 로그/구식 모델/캐시 등 소량 삭제(빠름)."""
    try:
        safe_cleanup.cleanup_logs_and_models(light=True)
        print(f"[CLEANUP] light done @ {_now().strftime('%H:%M:%S')}")
        sys.stdout.flush()
    except Exception as e:
        print(f"[CLEANUP] light error: {e}")
        sys.stdout.flush()

def _run_heavy():
    """무거운 모드: 공간 압박 시 실행. 시간이 다소 걸릴 수 있어 한가할 때만."""
    try:
        safe_cleanup.cleanup_logs_and_models(light=False)
        _mark_heavy_now()
        print(f"[CLEANUP] HEAVY done @ {_now().strftime('%H:%M:%S')}")
        sys.stdout.flush()
    except Exception as e:
        print(f"[CLEANUP] heavy error: {e}")
        sys.stdout.flush()

def _cleanup_job():
    """
    스케줄러에서 호출되는 본체.
    - busy(학습중/예측락/게이트닫힘/글로벌락)면: light-only 또는 skip
    - idle 이고 용량 압박 + 최소간격 충족: heavy
    """
    with _lock:
        try:
            used = safe_cleanup.get_directory_size_gb("/persistent")
        except Exception:
            used = -1.0

        busy = _is_training() or _is_predict_busy()
        print(f"[CLEANUP] tick busy={busy} used={used:.2f}GB "
              f"(soft={DISK_SOFTCAP_GB:.2f} hard={DISK_HARDCAP_GB:.2f})")
        sys.stdout.flush()

        # 하드캡 초과면 즉시 비상 정리 (safe_cleanup 내장 루틴 사용)
        try:
            if used >= DISK_HARDCAP_GB:
                print("[CLEANUP] EMERGENCY: hard cap exceeded -> run_emergency_purge()")
                sys.stdout.flush()
                safe_cleanup.run_emergency_purge()
                return
        except Exception:
            pass

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
    """외부에서 호출: 정리 스케줄러 시작 (중복 시작 방지)."""
    global _sched
    if _sched is not None:
        print("[CLEANUP] scheduler already running")
        sys.stdout.flush()
        return

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
    """외부에서 호출: 정리 스케줄러 완전 종료."""
    global _sched
    try:
        if _sched is not None:
            try:
                _sched.pause()
            except Exception:
                pass
            try:
                _sched.remove_all_jobs()
            except Exception:
                pass
            _sched.shutdown(wait=False)
            _sched = None
            print("[CLEANUP] scheduler stopped")
            sys.stdout.flush()
    except Exception as e:
        print(f"[CLEANUP] stop error: {e}")
        sys.stdout.flush()
