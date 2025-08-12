# training_memory.py (FINAL)

from collections import Counter
import sqlite3

# 필요 시 다른 곳에서 씀 (삭제해도 되지만 유지)
from failure_db import load_existing_failure_hashes  # noqa: F401

# ✅ 핵심 수정: wrong_data_loader의 함수를 별칭으로 임포트해서 재귀 제거
from wrong_data_loader import load_training_prediction_data as _wdl_load_training_prediction_data


def get_frequent_failures(min_count=5):
    counter = Counter()
    try:
        with sqlite3.connect("/persistent/logs/failure_patterns.db") as conn:
            rows = conn.execute("SELECT hash FROM failure_patterns").fetchall()
            for row in rows:
                counter[row[0]] += 1
    except Exception:
        return set()
    return {h for h, cnt in counter.items() if cnt >= min_count}


# ✅ 동일한 이름을 유지하되, 별칭으로 임포트한 실제 구현을 호출
#   - 원래 시그니처 유지 (source_type은 사용 안 하므로 **kwargs로 흡수)
def load_training_prediction_data(symbol, strategy, input_size, window, source_type="wrong", **kwargs):
    return _wdl_load_training_prediction_data(symbol, strategy, input_size, window, **kwargs)
