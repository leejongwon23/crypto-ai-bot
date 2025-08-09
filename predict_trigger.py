# === predict_trigger.py (최종본) ===
import os
import time
import traceback
import datetime
import pytz
import pandas as pd
import numpy as np
from collections import Counter

from data.utils import SYMBOLS, get_kline_by_strategy
from logger import log_audit_prediction as log_audit

# ──────────────────────────────────────────────────────────────
# 설정/상태
# ──────────────────────────────────────────────────────────────
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# 전략별 쿨다운(초)
TRIGGER_COOLDOWN = {"단기": 3600, "중기": 10800, "장기": 21600}
# 최근 예측 다양성 보정에 쓰일 모델 유형 표기 (필요 시 확장)
MODEL_TYPES = ["lstm", "cnn_lstm", "transformer"]

# 마지막 실행시각 기록
last_trigger_time = {}


# ──────────────────────────────────────────────────────────────
# 전조 조건점검
# ──────────────────────────────────────────────────────────────
def check_pre_burst_conditions(df: pd.DataFrame, strategy: str) -> bool:
    """
    변동성 확장·이평 압축·밴드 확장 등 간단 전조 시그널.
    데이터가 부족하면 기회를 주기 위해 True 반환(낙관적).
    """
    try:
        if df is None or len(df) < 10:
            print("[경고] 데이터 너무 적음 → fallback 조건 평가(True)")
            return True

        # 거래량 증가
        vol_increasing = False
        if len(df) >= 3 and "volume" in df.columns:
            vol_increasing = df["volume"].iloc[-3] < df["volume"].iloc[-2] < df["volume"].iloc[-1]

        # 최근 가격 안정
        price_slice = df["close"].iloc[-6:] if len(df) >= 6 else df["close"]
        stable_price = (price_slice.max() - price_slice.min()) / max(1e-9, price_slice.mean()) < 0.005

        # 이평 압축
        close = df["close"]
        ema_5 = close.ewm(span=5).mean().iloc[-1] if len(df) >= 5 else close.mean()
        ema_15 = close.ewm(span=15).mean().iloc[-1] if len(df) >= 15 else close.mean()
        ema_60 = close.ewm(span=60).mean().iloc[-1] if len(df) >= 60 else close.mean()
        ema_pack = max(ema_5, ema_15, ema_60) - min(ema_5, ema_15, ema_60)
        ema_compressed = ema_pack / max(1e-9, close.iloc[-1]) < 0.003

        # 밴드 확장(표준편차 증가)
        bb_std = close.pct_change().rolling(20).std() if len(df) >= 21 else pd.Series([0.0, 0.0])
        expanding_band = True
        if len(bb_std) >= 2 and bb_std.iloc[-2] > 0:
            expanding_band = (bb_std.iloc[-1] > bb_std.iloc[-2]) and (bb_std.iloc[-1] > 0.002)

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


# ──────────────────────────────────────────────────────────────
# 품질 필터 (차단용 X) — 항상 True
# ──────────────────────────────────────────────────────────────
def check_model_quality(symbol: str, strategy: str) -> bool:
    """
    모델 품질로 예측을 차단하지 않는다(메타러너가 자체적으로 반영).
    """
    return True


# ──────────────────────────────────────────────────────────────
# 트리거 런루프
# ──────────────────────────────────────────────────────────────
def run():
    """
    SYMBOLS × 전략별 전조 시그널을 점검하고, 조건 충족 시 예측을 실행.
    """
    from recommend import run_prediction  # 지연 임포트(순환참조 방지)

    print(f"[트리거 실행] 전조 패턴 감지 시작: {now_kst().isoformat()}")
    triggered = 0

    for symbol in SYMBOLS:
        for strategy in ["단기", "중기", "장기"]:
            try:
                key = f"{symbol}_{strategy}"
                now_ts = time.time()
                cooldown = TRIGGER_COOLDOWN.get(strategy, 3600)

                # 쿨다운
                if now_ts - last_trigger_time.get(key, 0) < cooldown:
                    print(f"[쿨다운] {key} 최근 실행됨 → 스킵")
                    continue

                # 데이터 확보
                df = get_kline_by_strategy(symbol, strategy)
                if df is None or len(df) < 60 or "close" not in df.columns:
                    print(f"[⛔ 데이터 부족] {symbol}-{strategy} → {len(df) if df is not None else 0}개")
                    continue

                # 전조 시그널 점검 (품질 필터는 항상 통과)
                if check_pre_burst_conditions(df, strategy) and check_model_quality(symbol, strategy):
                    print(f"[✅ 트리거 포착] {symbol} - {strategy} → 예측 실행")
                    try:
                        run_prediction(symbol, strategy)
                        last_trigger_time[key] = now_ts
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


# ──────────────────────────────────────────────────────────────
# 최근 클래스 분포 유틸
# ──────────────────────────────────────────────────────────────
def get_recent_class_frequencies(strategy: str = None, recent_days: int = 3) -> Counter:
    """
    최근 prediction_log에서 클래스 빈도 계산 (전략별 필터 가능)
    """
    try:
        path = "/persistent/logs/prediction_log.csv"
        if not os.path.exists(path):
            return Counter()

        df = pd.read_csv(path, encoding="utf-8-sig")
        if "predicted_class" not in df.columns:
            print("[⚠️ get_recent_class_frequencies] 'predicted_class' 컬럼 없음 → 빈 Counter 반환")
            return Counter()

        if strategy:
            df = df[df["strategy"] == strategy]

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        cutoff = pd.Timestamp.now(tz=pytz.UTC) - pd.Timedelta(days=recent_days)
        # 타임존 보정
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        df = df[df["timestamp"] >= cutoff]

        return Counter(df["predicted_class"].dropna().astype(int))

    except Exception as e:
        print(f"[⚠️ get_recent_class_frequencies 예외] {e}")
        return Counter()


# ──────────────────────────────────────────────────────────────
# 다양성 가중 확률 보정
# ──────────────────────────────────────────────────────────────
def adjust_probs_with_diversity(
    probs: np.ndarray,
    recent_freq: Counter,
    class_counts: dict = None,
    alpha: float = 0.10,
    beta: float = 0.10,
) -> np.ndarray:
    """
    probs: (C,) 또는 (1,C) ndarray
    recent_freq: 최근 예측 클래스 Counter
    class_counts: (선택) 클래스별 학습 분포
    alpha: 최근 과출현 클래스 패널티 강도
    beta: 데이터 부족 클래스 보상 강도
    """
    p = probs.copy()
    if p.ndim == 2:
        p = p[0]

    C = len(p)
    total_recent = sum(recent_freq.values()) + 1e-6

    # 최근 빈도 기반 weight (과출현 클래스 패널티)
    recent_weights = np.array([
        np.exp(-alpha * (recent_freq.get(i, 0) / total_recent))
        for i in range(C)
    ])
    recent_weights = np.clip(recent_weights, 0.85, 1.15)

    # 데이터 분포 기반 weight (희소 클래스 보상)
    if class_counts:
        total_class = sum(class_counts.values()) + 1e-6
        class_weights = np.array([
            np.exp(beta * (1.0 - class_counts.get(str(i), 0) / total_class))
            for i in range(C)
        ])
    else:
        class_weights = np.exp(np.ones(C) * beta)

    class_weights = np.clip(class_weights, 0.85, 1.15)

    combined = np.clip(recent_weights * class_weights, 0.85, 1.15)
    adjusted = p * combined
    adjusted /= max(1e-9, adjusted.sum())
    return adjusted


# ──────────────────────────────────────────────────────────────
# 모듈 단독 실행용
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
