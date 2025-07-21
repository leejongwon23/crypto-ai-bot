# safe_cleanup.py

import os
import time
from datetime import datetime, timedelta

# ✅ 삭제 대상 경로
LOG_DIR = "/persistent/logs"
DELETED_LOG_PATH = os.path.join(LOG_DIR, "deleted_log.txt")

# ✅ 보존 기간 (최근 3일치 로그는 유지)
KEEP_DAYS = 3

# ✅ 삭제 대상 접두어
DELETE_PREFIXES = [
    "prediction_",
    "evaluation_",
    "wrong_"
]

# ✅ 삭제 제외 파일명
EXCLUDE_FILES = set([
    "prediction_log.csv",
    "train_log.csv",
    "evaluation_result.csv",
    "deleted_log.txt",
    "wrong_predictions.csv",
    "fine_tune_target.csv",
])

# ✅ 실행 함수
def auto_delete_old_logs():
    now = datetime.now()
    cutoff = now - timedelta(days=KEEP_DAYS)
    deleted = []

    if not os.path.exists(LOG_DIR):
        print(f"[❌ 로그 디렉토리 없음] → {LOG_DIR}")
        return

    for fname in os.listdir(LOG_DIR):
        fpath = os.path.join(LOG_DIR, fname)

        if not os.path.isfile(fpath):
            continue
        if fname in EXCLUDE_FILES:
            continue
        if not any(fname.startswith(p) for p in DELETE_PREFIXES):
            continue

        try:
            # ✅ 날짜가 포함된 로그 파일만 타겟팅
            date_str = fname.split("_")[-1].replace(".csv", "").strip()
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date < cutoff:
                os.remove(fpath)
                deleted.append(fname)
        except Exception as e:
            print(f"[⚠️ 삭제 실패] {fname} → {e}")

    if deleted:
        with open(DELETED_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 삭제된 파일 목록:\n")
            for name in deleted:
                f.write(f"  - {name}\n")
        print(f"[🧹 삭제 완료] 총 {len(deleted)}개 파일 삭제됨.")
    else:
        print("[✅ 삭제 대상 없음] 최근 로그만 존재합니다.")

# ✅ main.py에서 import 시 자동 실행
auto_delete_old_logs()
