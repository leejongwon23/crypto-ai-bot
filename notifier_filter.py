# notifier_filter.py (FINAL aligned)
import os
import pandas as pd

# ✅ logger.py가 경로 단일화(get_PREDICTION_LOG_PATH) 쓰는 구조면 그걸 우선 사용
try:
    from config import get_PREDICTION_LOG_PATH
    PREDICTION_LOG_PATH = get_PREDICTION_LOG_PATH()
except Exception:
    PREDICTION_LOG_PATH = "/persistent/prediction_log.csv"


def _to_bool_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype=bool)
    s_str = s.astype(str).str.strip().str.lower()
    return s_str.isin({"true", "1", "yes", "y", "success", "v_success"})


def _is_shadow_or_hold(df: pd.DataFrame) -> pd.Series:
    """
    ✅ 섀도우/보류/실패로그(예측실패 등)로 알림 성공률이 오염되는 것 방지
    """
    direction = df.get("direction")
    if direction is None:
        return pd.Series([False] * len(df), index=df.index)

    d = direction.astype(str)
    # 섀도우, 보류, 예측실패 등은 알림 성공률 통계에서 제외
    return (
        d.str.contains("섀도우", na=False)
        | d.str.contains("보류", na=False)
        | d.str.contains("예측실패", na=False)
    )


def _is_final_evaluated_status(df: pd.DataFrame) -> pd.Series:
    """
    ✅ 평가가 '완료된' 결과만 성공률 계산에 포함
    - success / fail / v_success / v_fail 만 인정
    - pending / invalid 등은 제외
    """
    st = df.get("status")
    if st is None:
        return pd.Series([False] * len(df), index=df.index)

    s = st.astype(str).str.strip().str.lower()
    return s.isin({"success", "fail", "v_success", "v_fail"})


def should_send_notification(
    symbol: str,
    strategy: str,
    model: str = "meta",
    *,
    min_success_rate: float = 0.60,   # ✅ 계획대로 60%
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

        # ✅ 섀도우/보류/예측실패 등 제외
        base = base[~_is_shadow_or_hold(base)]
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

        if sub.empty:
            return allow_first_time

        # ✅ 평가 완료된 결과만 포함
        if "status" in sub.columns:
            sub = sub[_is_final_evaluated_status(sub)]
            if len(sub) < int(min_samples):
                return allow_first_time

            st = sub["status"].astype(str).str.strip().str.lower()
            success_rate = st.isin(["success", "v_success", "true", "1"]).mean()
            return float(success_rate) >= float(min_success_rate)

        # status가 없으면 success 불린 컬럼 대안
        if "success" in sub.columns:
            # success 기반일 때도 샘플 수는 그대로 체크
            if len(sub) < int(min_samples):
                return allow_first_time
            success_rate = _to_bool_series(sub["success"]).mean()
            return float(success_rate) >= float(min_success_rate)

        # 둘 다 없으면 구조 유지
        return allow_first_time

    except Exception as e:
        print(f"[notifier_filter 예외] {e}")
        return allow_first_time
