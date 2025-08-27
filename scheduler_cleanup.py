# scheduler_cleanup.py
import os
import atexit
from typing import Optional
from pytz import timezone
from apscheduler.schedulers.background import BackgroundScheduler

# safe_cleanup 에서 공용 락/정리 함수 가져오기
import safe_cleanup
from safe_cleanup import cleanup_logs_and_models

_scheduler: Optional[BackgroundScheduler] = None
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
        cleanup_logs_and_models()
    except Exception as e:
        print(f"[cleanup] run failed: {e}", flush=True)


def start_cleanup_scheduler():
    """
    - 서비스 시작 시 1회 즉시 실행
    - 이후 SAFE_CLEANUP_INTERVAL_MIN(기본 30분)마다 자동 실행
    - 중복/동시 실행 방지 및 재시작 안전
    """
    global _scheduler
    if _scheduler is not None:
        # 이미 실행 중이면 그대로 반환
        if is_cleanup_scheduler_running():
            return _scheduler
        # 객체는 있으나 멈춰있으면 재시작
        try:
            _scheduler.start()
            return _scheduler
        except Exception:
            pass  # 아래에서 새로 만든다

    tz = timezone(os.getenv("TZ", "Asia/Seoul"))
    interval_min = int(os.getenv("SAFE_CLEANUP_INTERVAL_MIN", "30"))

    _scheduler = BackgroundScheduler(timezone=tz)
    # coalesce: 지연 시 1회로 합치기, max_instances=1: 동시 실행 방지
    _scheduler.add_job(
        _cleanup_job,
        trigger="interval",
        minutes=interval_min,
        id=_JOB_ID,
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=600,
    )

    # 부팅 직후 1회 즉시 실행(락이면 스킵)
    try:
        _cleanup_job()
    except Exception as e:
        print(f"[cleanup] startup run failed: {e}", flush=True)

    _scheduler.start()
    atexit.register(lambda: stop_cleanup_scheduler())
    print(f"[cleanup] scheduler started (every {interval_min} min)", flush=True)
    return _scheduler


def stop_cleanup_scheduler(wait: bool = False):
    """스케줄러를 안전하게 중지한다. 여러 번 호출해도 안전(idempotent)."""
    global _scheduler
    if _scheduler is None:
        return False
    try:
        # 등록된 작업 제거(보수적)
        try:
            _scheduler.remove_all_jobs()
        except Exception:
            pass
        _scheduler.shutdown(wait=wait)
        print("[cleanup] scheduler stopped", flush=True)
        return True
    except Exception as e:
        print(f"[cleanup] stop error: {e}", flush=True)
        return False
    finally:
        _scheduler = None


def is_cleanup_scheduler_running() -> bool:
    """실행 상태 조회."""
    try:
        return _scheduler is not None and _scheduler.state == 1  # STATE_RUNNING
    except Exception:
        return False


def run_cleanup_now():
    """수동으로 즉시 1회 실행 (락이면 스킵)."""
    _cleanup_job()
