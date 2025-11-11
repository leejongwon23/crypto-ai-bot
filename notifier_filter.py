# notifier_filter.py (FINAL aligned)
import os
import pandas as pd

PREDICTION_LOG_PATH = "/persistent/prediction_log.csv"

def _to_bool_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype=bool)
    s_str = s.astype(str).str.strip().str.lower()
    return s_str.isin({"true", "1", "yes", "y", "success", "v_success"})

def should_send_notification(
    symbol: str,
    strategy: str,
    model: str = "meta",
    *,
    min_success_rate: float = 0.60,   # ← 계획대로 60%
    min_samples: int = 10,
    allow_first_time: bool = False
) -> bool:
    try:
        if not os.path.exists(PREDICTION_LOG_PATH):
            return allow_first_time

        df = pd.read_csv(PREDICTION_LOG_PATH, encoding="utf-8-sig", on_bad_lines="skip")
        if df.empty:
            return allow_first_time

        # 기본 필터: symbol / strategy
        base = df[(df.get("symbol") == symbol) & (df.get("strategy") == strategy)]
        if base.empty:
            return allow_first_time

        # 1) model 컬럼 우선
        if "model" in base.columns:
            sub = base[base["model"] == model]
        else:
            sub = base

        # 2) 그래도 비면 model_name 에서도 한 번 더
        if sub.empty and "model_name" in base.columns:
            sub = base[base["model_name"].astype(str).str.startswith(model)]

        if len(sub) < min_samples:
            return allow_first_time

        # status 우선
        if "status" in sub.columns:
            st = sub["status"].astype(str).str.strip().str.lower()
            success_rate = st.isin(["success", "v_success", "true", "1"]).mean()
            return success_rate >= min_success_rate

        # success 불린 컬럼 대안
        if "success" in sub.columns:
            success_rate = _to_bool_series(sub["success"]).mean()
            return success_rate >= min_success_rate

        # 둘 다 없으면 구조 유지
        return allow_first_time

    except Exception as e:
        print(f"[notifier_filter 예외] {e}")
        return allow_first_time
