# notifier_filter.py (FINAL)
import os
import pandas as pd

PREDICTION_LOG_PATH = "/persistent/prediction_log.csv"

def _to_bool_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype=bool)
    s_str = s.astype(str).str.strip().str.lower()
    return s_str.isin({"true", "1", "yes", "y"})

def should_send_notification(
    symbol: str,
    strategy: str,
    model: str = "meta",
    *,
    min_success_rate: float = 0.65,
    min_samples: int = 10,
    allow_first_time: bool = False
) -> bool:
    try:
        if not os.path.exists(PREDICTION_LOG_PATH):
            return allow_first_time

        df = pd.read_csv(PREDICTION_LOG_PATH, encoding="utf-8-sig", on_bad_lines="skip")
        if df.empty:
            return allow_first_time

        need_cols = {"symbol", "strategy", "model"}
        if not need_cols.issubset(df.columns):
            return allow_first_time

        sub = df[(df["symbol"] == symbol) & (df["strategy"] == strategy) & (df["model"] == model)]
        if len(sub) < min_samples:
            return allow_first_time

        if "status" in sub.columns:
            st = sub["status"].astype(str).str.strip().str.lower()
            success_rate = st.isin(["success", "v_success"]).mean()
            return success_rate >= min_success_rate

        if "success" in sub.columns:
            success_rate = _to_bool_series(sub["success"]).mean()
            return success_rate >= min_success_rate

        return allow_first_time

    except Exception as e:
        print(f"[notifier_filter 예외] {e}")
        return allow_first_time
