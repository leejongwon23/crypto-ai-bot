# notifier_filter.py
import os
import pandas as pd

PREDICTION_LOG_PATH = "/persistent/prediction_log.csv"

def should_send_notification(symbol, strategy, model="meta",
                              min_success_rate=0.65, min_samples=10,
                              allow_first_time=False):
    """
    텔레그램 전송 여부를 판단하는 필터 함수
    - symbol: 종목명
    - strategy: 전략명 (단기/중기/장기)
    - model: 모델명 (기본값: meta)
    - min_success_rate: 최소 성공률 기준
    - min_samples: 최소 예측 횟수 기준
    - allow_first_time: 첫 예측도 허용할지 여부
    """
    try:
        if not os.path.exists(PREDICTION_LOG_PATH):
            return allow_first_time

        df = pd.read_csv(PREDICTION_LOG_PATH, encoding="utf-8-sig")
        if not {"symbol", "strategy", "model", "success"}.issubset(df.columns):
            # 필요한 컬럼이 없으면 첫 예측 여부로만 판단
            return allow_first_time

        # 해당 심볼+전략+모델의 과거 예측만 필터링
        sub = df[(df["symbol"] == symbol) &
                 (df["strategy"] == strategy) &
                 (df["model"] == model)]

        if len(sub) < min_samples:
            return allow_first_time

        # 성공률 계산
        success_rate = sub["success"].mean()

        # 기준 충족 여부 반환
        return success_rate >= min_success_rate

    except Exception as e:
        print(f"[notifier_filter 예외] {e}")
        return allow_first_time
