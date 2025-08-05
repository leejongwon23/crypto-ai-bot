import pandas as pd
import time
import traceback
import datetime
import pytz

from data.utils import SYMBOLS, get_kline_by_strategy
from logger import log_audit_prediction as log_audit

last_trigger_time = {}
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
TRIGGER_COOLDOWN = {"단기": 3600, "중기": 10800, "장기": 21600}
MODEL_TYPES = ["lstm", "cnn_lstm", "transformer"]

def check_pre_burst_conditions(df, strategy):
    try:
        if df is None or len(df) < 10:
            print("[경고] 데이터 너무 적음 → fallback 조건 평가")
            # ✅ 데이터가 부족해도 기회를 주기 위해 True 반환
            return True

        vol_increasing = df['volume'].iloc[-3] < df['volume'].iloc[-2] < df['volume'].iloc[-1]
        price_range = df['close'].iloc[-6:]
        stable_price = (price_range.max() - price_range.min()) / price_range.mean() < 0.005

        ema_5 = df['close'].ewm(span=5).mean().iloc[-1] if len(df) >= 5 else df['close'].mean()
        ema_15 = df['close'].ewm(span=15).mean().iloc[-1] if len(df) >= 15 else df['close'].mean()
        ema_60 = df['close'].ewm(span=60).mean().iloc[-1] if len(df) >= 60 else df['close'].mean()
        ema_pack = max(ema_5, ema_15, ema_60) - min(ema_5, ema_15, ema_60)
        ema_compressed = ema_pack / df['close'].iloc[-1] < 0.003

        bb_std = df['close'].rolling(window=20).std() if len(df) >= 20 else pd.Series([0.0])
        expanding_band = bb_std.iloc[-2] < bb_std.iloc[-1] and bb_std.iloc[-1] > 0.002 if len(bb_std) >= 2 else True

        if strategy == "단기":
            return sum([vol_increasing, stable_price, ema_compressed, expanding_band]) >= 2
        elif strategy == "중기":
            return sum([stable_price, ema_compressed, expanding_band]) >= 2
        elif strategy == "장기":
            return sum([ema_compressed, expanding_band]) >= 1
        else:
            return False
    except Exception as e:
        print(f"[조건 점검 오류] {e}")
        traceback.print_exc()
        return False

# ✅ 품질 필터 제거 버전
def check_model_quality(symbol, strategy):
    """
    모델 품질 필터 제거
    성공률은 메타러너 참고용만 유지하며,
    예측 차단에는 사용하지 않음.
    """
    return True

def run():
    from recommend import run_prediction  # ✅ 순환참조 방지

    print(f"[트리거 실행] 전조 패턴 감지 시작: {now_kst().isoformat()}")
    triggered = 0  # ✅ 실행 횟수 기록

    for symbol in SYMBOLS:
        for strategy in ["단기", "중기", "장기"]:
            try:
                key = f"{symbol}_{strategy}"
                now = time.time()
                cooldown = TRIGGER_COOLDOWN.get(strategy, 3600)

                if now - last_trigger_time.get(key, 0) < cooldown:
                    print(f"[쿨다운] {key} 최근 실행됨 → 스킵")
                    continue

                df = get_kline_by_strategy(symbol, strategy)
                if df is None or len(df) < 60:
                    print(f"[⛔ 데이터 부족] {symbol}-{strategy} → {len(df) if df is not None else 0}개")
                    continue

                # ✅ 품질 필터 제거됨 → 모든 경우 예측 시도
                if check_pre_burst_conditions(df, strategy):
                    print(f"[✅ 트리거 포착] {symbol} - {strategy} → 예측 실행")

                    try:
                        run_prediction(symbol, strategy, source="변동성")
                        last_trigger_time[key] = now
                        log_audit(symbol, strategy, "트리거예측", "조건 만족으로 실행")
                        triggered += 1

                    except Exception as inner:
                        print(f"[❌ 예측 실행 실패] {symbol}-{strategy}: {inner}")
                        log_audit(symbol, strategy, "트리거예측오류", f"예측실행실패: {inner}")
                else:
                    print(f"[조건 미충족] {symbol}-{strategy}")

            except Exception as e:
                print(f"[트리거 오류] {symbol} {strategy}: {e}")
                log_audit(symbol, strategy or "알수없음", "트리거오류", str(e))

    print(f"🔁 이번 트리거 루프에서 예측 실행된 개수: {triggered}")

# ✅ 최근 클래스 빈도 계산
from collections import Counter
import os
from datetime import datetime as dt

def get_recent_class_frequencies(strategy=None, recent_days=3):
    from collections import Counter
    import os
    import pandas as pd

    try:
        path = "/persistent/prediction_log.csv"
        if not os.path.exists(path):
            return Counter()

        df = pd.read_csv(path, encoding="utf-8-sig")
        if "predicted_class" not in df.columns:
            print("[⚠️ get_recent_class_frequencies] 'predicted_class' 컬럼 없음 → 빈 Counter 반환")
            return Counter()

        if strategy:
            df = df[df["strategy"] == strategy]

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
        df = df[df["timestamp"] >= cutoff]

        return Counter(df["predicted_class"].dropna().astype(int))

    except Exception as e:
        print(f"[⚠️ get_recent_class_frequencies 예외] {e}")
        return Counter()

# ✅ 확률 보정 함수
import numpy as np
from collections import Counter

def adjust_probs_with_diversity(probs, recent_freq: Counter, class_counts: dict = None, alpha=0.10, beta=0.10):
    import numpy as np

    probs = probs.copy()
    if probs.ndim == 2:
        probs = probs[0]

    num_classes = len(probs)
    total_recent = sum(recent_freq.values()) + 1e-6

    # ✅ 최근 빈도 기반 weight
    recent_weights = np.array([
        np.exp(-alpha * (recent_freq.get(i, 0) / total_recent))
        for i in range(num_classes)
    ])
    recent_weights = np.clip(recent_weights, 0.85, 1.15)

    if class_counts:
        total_class = sum(class_counts.values()) + 1e-6
        class_weights = np.array([
            np.exp(beta * (1.0 - class_counts.get(str(i), 0) / total_class))
            for i in range(num_classes)
        ])
    else:
        class_weights = np.exp(np.ones(num_classes) * beta)

    class_weights = np.clip(class_weights, 0.85, 1.15)

    combined_weights = recent_weights * class_weights
    combined_weights = np.clip(combined_weights, 0.85, 1.15)

    adjusted = probs * combined_weights
    adjusted /= adjusted.sum()
    return adjusted
