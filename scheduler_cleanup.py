# scheduler_cleanup.py (FINAL — deduped, compatible wrapper)
import os
import atexit
import threading
from typing import Optional

# 공용 정리 로직/락 경로
import safe_cleanup
from safe_cleanup import cleanup_logs_and_models  # 호환을 위해 유지

# ─────────────────────────────────────────────────────────────
# 실행 모드 결정
#   - 기본: safe_cleanup 내부 스케줄러 스레드 사용(중복 제거)
#   - 환경변수 USE_APSCHEDULER=1 이면 APScheduler 모드 강제
# ─────────────────────────────────────────────────────────────
_USE_APS = os.getenv("USE_APSCHEDULER", "0").strip() == "1"

# APS 모드 준비(없으면 자동으로 래퍼 모드로 폴백)
BackgroundScheduler = None
if _USE_APS:
    try:
        from pytz import timezone
        from apscheduler.schedulers.background import BackgroundScheduler as _BG
        BackgroundScheduler = _BG
    except Exception:
        # 라이브러리 없으면 자동 폴백
        _USE_APS = False

# 공통 상태
_scheduler_thread: Optional[threading.Thread] = None  # 래퍼 모드용
_scheduler_aps: Optional["BackgroundScheduler"] = None  # APS 모드용
_JOB_ID = "cleanup_job"


def _should_skip() -> bool:
    """글로벌 락이 잡혀 있으면 정리 작업을 스킵한다."""
    lock_path = getattr(safe_cleanup, "LOCK_PATH", None)
    if lock_path and os.path.exists(lock_path):
        print("[cleanup] lock detected → skip this run", flush=True)
        return True
    return False


def _cleanup_job():
    """스케줄러가 호출하는 실제 작업 (락 체크 포함)."""
    try:
        if _should_skip():
            return
        # safe_cleanup 내부가 용량/보호시간/모델세트 정책을 처리
        cleanup_logs_and_models()
    except Exception as e:
        print(f"[cleanup] run failed: {e}", flush=True)


# ─────────────────────────────────────────────────────────────
# 래퍼 모드 (기본): safe_cleanup 내부 스레드 스케줄러 사용
# ─────────────────────────────────────────────────────────────
def _start_wrapper_mode():
    global _scheduler_thread
    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        return _scheduler_thread
    # safe_cleanup 안의 스케줄러는 daemon 스레드로 주기 실행
    _scheduler_thread = safe_cleanup.start_cleanup_scheduler(daemon=True)
    atexit.register(lambda: stop_cleanup_scheduler())  # 호환용 로그만 남김
    print("[cleanup] wrapper mode started (safe_cleanup internal scheduler).", flush=True)
    return _scheduler_thread


# ─────────────────────────────────────────────────────────────
# APS 모드 (선택): APScheduler 사용 — 중복 방지 옵션 포함
# ─────────────────────────────────────────────────────────────
def _start_aps_mode():
    global _scheduler_aps
    if _scheduler_aps is not None:
        # 이미 실행 중이면 그대로 반환
        if is_cleanup_scheduler_running():
            return _scheduler_aps
        try:
            _scheduler_aps.start()
            return _scheduler_aps
        except Exception:
            pass  # 새로 생성

    tz = timezone(os.getenv("TZ", "Asia/Seoul"))
    interval_min = int(os.getenv("SAFE_CLEANUP_INTERVAL_MIN", "30"))

    _scheduler_aps = BackgroundScheduler(timezone=tz)
    _scheduler_aps.add_job(
        _cleanup_job,
        trigger="interval",
        minutes=interval_min,
        id=_JOB_ID,
        replace_existing=True,
        coalesce=True,           # 지연 시 1회로 합치기
        max_instances=1,         # 동시 실행 방지
        misfire_grace_time=600,
    )

    # 부팅 직후 1회 즉시 실행(락이면 스킵)
    try:
        _cleanup_job()
    except Exception as e:
        print(f"[cleanup] startup run failed: {e}", flush=True)

    _scheduler_aps.start()
    atexit.register(lambda: stop_cleanup_scheduler())
    print(f"[cleanup] APS mode started (every {interval_min} min)", flush=True)
    return _scheduler_aps


# ─────────────────────────────────────────────────────────────
# 공개 API (인터페이스 유지)
# ─────────────────────────────────────────────────────────────
def start_cleanup_scheduler():
    """
    - 기본: safe_cleanup 내부 스케줄러 스레드 1회만 기동(중복 제거).
    - USE_APSCHEDULER=1 이면 APScheduler 모드로 실행.
    """
    if _USE_APS:
        return _start_aps_mode()
    return _start_wrapper_mode()


def stop_cleanup_scheduler(wait: bool = False):
    """
    스케줄러 중지.
    - 래퍼 모드: safe_cleanup 내부 루프는 종료 훅이 없어 실제 중지는 불가(호환용 로그만 출력).
    - APS 모드: 정상적으로 shutdown.
    """
    global _scheduler_thread, _scheduler_aps
    if _USE_APS:
        if _scheduler_aps is None:
            return False
        try:
            try:
                _scheduler_aps.remove_all_jobs()
            except Exception:
                pass
            _scheduler_aps.shutdown(wait=wait)
            print("[cleanup] APS scheduler stopped", flush=True)
            return True
        except Exception as e:
            print(f"[cleanup] stop error: {e}", flush=True)
            return False
        finally:
            _scheduler_aps = None
    else:
        print("[cleanup] wrapper mode: stop not supported (safe_cleanup runs a persistent daemon).", flush=True)
        _scheduler_thread = None
        return False


def is_cleanup_scheduler_running() -> bool:
    """실행 상태 조회."""
    if _USE_APS:
        try:
            # 1 == STATE_RUNNING
            return _scheduler_aps is not None and getattr(_scheduler_aps, "state", 0) == 1
        except Exception:
            return False
    try:
        return _scheduler_thread is not None and _scheduler_thread.is_alive()
    except Exception:
        return False


def run_cleanup_now():
    """수동으로 즉시 1회 실행 (락이면 스킵)."""
    _cleanup_job()
