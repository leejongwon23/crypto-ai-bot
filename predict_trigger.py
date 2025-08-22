# === predict_trigger.py (MEM-SAFE FINAL) ===
import os
import pandas as pd
import time
import traceback
import datetime
import pytz
from collections import Counter
import numpy as np

from data.utils import SYMBOLS, get_kline_by_strategy
from logger import log_audit_prediction as log_audit, ensure_prediction_log_exists

# ▷ (옵션) 레짐/캘리브레이션: 없으면 안전 통과
try:
    from regime_detector import detect_regime
except Exception:
    def detect_regime(symbol, strategy, now=None):
        return "unknown"

try:
    from calibration import get_calibration_version
except Exception:
    def get_calibration_version():
        return "none"

# ===== 설정(환경변수로 조절 가능) =====
TRIGGER_COOLDOWN = {"단기": 3600, "중기": 10800, "장기": 21600}
MODEL_TYPES = ["lstm", "cnn_lstm", "transformer"]
MAX_LOOKBACK = int(os.getenv("TRIGGER_MAX_LOOKBACK", "180"))   # 전조 계산시 최근 N행만 사용
RECENT_DAYS_FOR_FREQ = int(os.getenv("TRIGGER_FREQ_DAYS", "3"))
CSV_CHUNKSIZE = int(os.getenv("TRIGGER_CSV_CHUNKSIZE", "50000"))

last_trigger_time = {}
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# ──────────────────────────────────────────────────────────────
# 전조 조건
# ──────────────────────────────────────────────────────────────
def check_pre_burst_conditions(df, strategy):
    try:
        if df is None or len(df) < 10:
            print("[경고] 데이터 너무 적음 → fallback 조건 평가")
            return True

        # 메모리/연산량 절약: 최근 구간만 사용
        if MAX_LOOKBACK > 0 and len(df) > MAX_LOOKBACK:
            df = df.tail(MAX_LOOKBACK)

        vol_increasing = df['volume'].iloc[-3] < df['volume'].iloc[-2] < df['volume'].iloc[-1]
        price_range = df['close'].iloc[-6:]
        stable_price = (price_range.max() - price_range.min()) / (price_range.mean() + 1e-12) < 0.005

        ema_5 = df['close'].ewm(span=5).mean().iloc[-1] if len(df) >= 5 else df['close'].mean()
        ema_15 = df['close'].ewm(span=15).mean().iloc[-1] if len(df) >= 15 else df['close'].mean()
        ema_60 = df['close'].ewm(span=60).mean().iloc[-1] if len(df) >= 60 else df['close'].mean()
        ema_pack = max(ema_5, ema_15, ema_60) - min(ema_5, ema_15, ema_60)
        ema_compressed = ema_pack / (df['close'].iloc[-1] + 1e-12) < 0.003

        if len(df) >= 20:
            bb_std = df['close'].rolling(window=20).std()
            expanding_band = (bb_std.iloc[-2] < bb_std.iloc[-1]) and (bb_std.iloc[-1] > 0.002)
        else:
            expanding_band = True

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

def check_model_quality(symbol, strategy):
    return True

# ──────────────────────────────────────────────────────────────
# 트리거 실행 루프
# ──────────────────────────────────────────────────────────────
def run():
    try:
        from predict import predict as _predict
    except Exception as e:
        print(f"[치명] predict 모듈 로드 실패 → 트리거 중단: {e}")
        traceback.print_exc()
        return

    try:
        ensure_prediction_log_exists()
    except Exception as e:
        print(f"[경고] prediction_log 보장 실패: {e}")

    print(f"[트리거 실행] 전조 패턴 감지 시작: {now_kst().isoformat()}")
    triggered = 0

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
                    print(f"[⛔ 데이터 부족] {symbol}-{strategy} → {len(df) if isinstance(df, pd.DataFrame) else 0}개")
                    continue

                if check_pre_burst_conditions(df, strategy):
                    # 프리로드(로그용)
                    try:
                        regime = detect_regime(symbol, strategy, now=now_kst())
                        calib_ver = get_calibration_version()
                        log_audit(symbol, strategy, "프리로드", f"regime={regime}, calib_ver={calib_ver}")
                    except Exception as preload_e:
                        print(f"[프리로드 경고] {symbol}-{strategy}: {preload_e}")

                    print(f"[✅ 트리거 포착] {symbol} - {strategy} → 예측 실행")
                    try:
                        _predict(symbol, strategy, source="변동성")
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

# ──────────────────────────────────────────────────────────────
# 최근 클래스 빈도(메모리 안전: 청크 누산)
# ──────────────────────────────────────────────────────────────
def get_recent_class_frequencies(strategy=None, recent_days=RECENT_DAYS_FOR_FREQ):
    path = "/persistent/prediction_log.csv"
    if not os.path.exists(path):
        return Counter()

    cutoff = pd.Timestamp.now(tz="Asia/Seoul") - pd.Timedelta(days=int(recent_days))
    cols = ["timestamp", "strategy", "predicted_class"]
    freq = Counter()

    try:
        for chunk in pd.read_csv(path, usecols=lambda c: c in cols, encoding="utf-8-sig",
                                 chunksize=CSV_CHUNKSIZE, on_bad_lines="skip"):
            if "timestamp" not in chunk.columns or "predicted_class" not in chunk.columns:
                continue
            if strategy:
                chunk = chunk[chunk["strategy"] == strategy]

            # ts 파싱 최소화
            ts = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
            # KST 변환(naive도 안전히 처리)
            ts = ts.dt.tz_convert("Asia/Seoul")
            mask = ts >= cutoff
            if not mask.any():
                continue

            sub = chunk.loc[mask, "predicted_class"].dropna()
            try:
                vals = sub.astype(int).tolist()
            except Exception:
                vals = [int(float(x)) for x in sub if str(x).strip() != ""]
            freq.update(vals)
        return freq
    except Exception as e:
        print(f"[⚠️ get_recent_class_frequencies 예외] {e}")
        return Counter()

# ──────────────────────────────────────────────────────────────
def adjust_probs_with_diversity(probs, recent_freq: Counter, class_counts: dict = None, alpha=0.10, beta=0.10):
    p = probs.copy()
    if p.ndim == 2:
        p = p[0]
    num_classes = len(p)
    total_recent = sum(recent_freq.values()) + 1e-6

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
    combined = np.clip(recent_weights * class_weights, 0.85, 1.15)
    adjusted = p * combined
    s = adjusted.sum()
    if s <= 0:
        return p
    return adjusted / s
