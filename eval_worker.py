# === eval_worker.py (v2025-10-26) ===========================================
# 사용법:
#   python eval_worker.py            # 단독 실행
# 또는 앱 시작 시
#   from eval_worker import start_eval_loop
#   start_eval_loop()                # 백그라운드 스레드
# ===========================================================================
import os, time, threading, traceback
from typing import Callable, Dict, Any, List

from eval_queue import (
    fetch_batch, mark_done, mark_failed,
    reset_stuck, get_env_batch_size, get_env_stuck_min
)

# ✳️ 여기만 프로젝트에 맞게 연결하면 된다.
# 예측 1건(payload 포함)을 평가하고, 내부에서 관우로그/실패로그 등 기록하는 함수로 바꿔줘.
def evaluate_one(item: Dict[str, Any]) -> None:
    """
    item = {
      "id": int, "symbol": str, "strategy": str,
      "payload": dict(예측원본/확률 등)
    }
    ↓↓↓ 아래를 너의 평가 로직으로 교체하세요 ↓↓↓
    """
    # 예: 기존 평가 모듈이 있다면 이렇게 부른다:
    # from evaluator import evaluate_prediction
    # evaluate_prediction(item["symbol"], item["strategy"], item["payload"])
    #
    # 임시 기본 동작: 아무것도 안 하고 통과
    return

def process_once(batch_size: int | None = None) -> int:
    bs = batch_size or get_env_batch_size(100)
    jobs = fetch_batch(limit=bs)
    if not jobs:
        return 0
    ok_ids: List[int] = []
    fail_ids: List[int] = []
    for j in jobs:
        try:
            evaluate_one(j)  # ← 실제 평가
            ok_ids.append(j["id"])
        except Exception:
            fail_ids.append(j["id"])
            print(f"[eval fail] id={j['id']} {traceback.format_exc()}")
    if ok_ids:
        mark_done(ok_ids)
    if fail_ids:
        # 짧게 쉬었다가 재시도(기본 60초 뒤)
        retry_delay = int(os.getenv("EVAL_RETRY_DELAY_SEC", "60"))
        mark_failed(fail_ids, retry_delay_s=retry_delay)
    return len(jobs)

def run_forever():
    stuck_min = get_env_stuck_min(15)
    idle_ms = int(os.getenv("EVAL_IDLE_MS", "500"))
    while True:
        try:
            # 가끔씩 러닝 상태 복구
            reset_stuck(max_minutes=stuck_min)
            n = process_once()
            if n == 0:
                time.sleep(idle_ms / 1000.0)
        except Exception as e:
            print(f"[eval loop] err: {e}")
            time.sleep(1.0)

# 앱에서 쉽게 켜기 위한 스레드 버전
_THREAD = None
def start_eval_loop() -> bool:
    global _THREAD
    if _THREAD and _THREAD.is_alive():
        return False
    t = threading.Thread(target=run_forever, daemon=True)
    t.start()
    _THREAD = t
    return True

if __name__ == "__main__":
    run_forever()
