# scheduler_cleanup.py
import os
import atexit
from pytz import timezone
from apscheduler.schedulers.background import BackgroundScheduler

from safe_cleanup import cleanup_logs_and_models

_scheduler = None

def start_cleanup_scheduler():
    """
    - 서비스 시작 시 1회 즉시 실행
    - 이후 SAFE_CLEANUP_INTERVAL_MIN(기본 30분)마다 자동 실행
    - 중복/동시 실행 방지
    """
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    tz = timezone(os.getenv("TZ", "Asia/Seoul"))
    interval_min = int(os.getenv("SAFE_CLEANUP_INTERVAL_MIN", "30"))

    _scheduler = BackgroundScheduler(timezone=tz, daemon=True)
    _scheduler.add_job(
        cleanup_logs_and_models,
        trigger="interval",
        minutes=interval_min,
        id="cleanup_job",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=600
    )

    try:
        cleanup_logs_and_models()
    except Exception as e:
        print(f"[cleanup] startup run failed: {e}", flush=True)

    _scheduler.start()
    atexit.register(lambda: _scheduler.shutdown(wait=False))
    print(f"[cleanup] scheduler started (every {interval_min} min)", flush=True)
    return _scheduler
