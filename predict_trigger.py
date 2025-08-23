# === predict_trigger.py (MEM-SAFE FINAL — drop‑in, safer freqs & diversity, unified symbols) ===
import os
import pandas as pd
import time
import traceback
import datetime
import pytz
from collections import Counter
import numpy as np

# ✅ 심볼은 항상 data.utils 단일 소스에서 — stale 방지
from data.utils import get_ALL_SYMBOLS, get_kline_by_strategy
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
RECENT_DAYS_FOR_FREQ = max(1, int(os.getenv("TRIGGER_FREQ_DAYS", "3")))
CSV_CHUNKSIZE = max(10000, int(os.getenv("TRIGGER_CSV_CHUNKSIZE", "50000")))

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

    for symbol in get_ALL_SYMBOLS():
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
# 최근 클래스 빈도(메모리 안전: 청크 누산, 빈 로그/누락 컬럼/타임존 안전)
# ──────────────────────────────────────────────────────────────
def get_recent_class_frequencies(strategy=None, recent_days=RECENT_DAYS_FOR_FREQ):
    path = "/persistent/prediction_log.csv"
    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        return Counter()

    cutoff = pd.Timestamp.now(tz="Asia/Seoul") - pd.Timedelta(days=int(max(1, recent_days)))
    need = {"timestamp", "predicted_class", "strategy"}
    freq = Counter()

    try:
        for chunk in pd.read_csv(
            path,
            usecols=lambda c: (c in need) or (c == "predicted_class"),
            encoding="utf-8-sig",
            chunksize=CSV_CHUNKSIZE,
            on_bad_lines="skip"
        ):
            # 필수 컬럼 방어
            if "predicted_class" not in chunk.columns or "timestamp" not in chunk.columns:
                continue
            # 전략 필터(없으면 전체)
            if strategy and "strategy" in chunk.columns:
                chunk = chunk[chunk["strategy"] == strategy]

            # timestamp 파싱(utc 기준) → KST
            ts = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
            # tz가 없는 값이 섞여 있어도 안전 변환
            try:
                ts = ts.dt.tz_convert("Asia/Seoul")
            except Exception:
                ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")

            mask = ts >= cutoff
            if not mask.any():
                continue

            # 클래스 정수화 & 음수/결측 제거
            sub = chunk.loc[mask, "predicted_class"].dropna()
            vals = []
            for x in sub:
                try:
                    v = int(float(x))
                    if v >= 0:
                        vals.append(v)
                except Exception:
                    continue
            if vals:
                freq.update(vals)

        return freq
    except Exception as e:
        print(f"[⚠️ get_recent_class_frequencies 예외] {e}")
        return Counter()

# ──────────────────────────────────────────────────────────────
# 확률 보정: 최근 과다/과소 예측 및 클래스 불균형을 완만히 보정 (빈 입력/음수/NaN 모두 안전)
# ──────────────────────────────────────────────────────────────
def adjust_probs_with_diversity(probs, recent_freq: Counter, class_counts: dict = None, alpha=0.10, beta=0.10):
    # p을 1D float 배열로 정리하고, 음수/NaN 클리핑
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 2:
        p = p[0]
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.clip(p, 0.0, None)
    s0 = p.sum()
    if s0 <= 0:
        # 완전 비정상 입력이면 균등 분포 반환
        return np.ones_like(p) / max(1, len(p))
    p = p / s0

    num_classes = len(p)

    # 최근 빈도 가중치 (없으면 1.0로 처리)
    total_recent = float(sum(recent_freq.values()))
    if total_recent <= 0:
        recent_weights = np.ones(num_classes, dtype=np.float64)
    else:
        recent_weights = np.array([
            np.exp(-alpha * (float(recent_freq.get(i, 0)) / total_recent))
            for i in range(num_classes)
        ], dtype=np.float64)
        recent_weights = np.clip(recent_weights, 0.85, 1.15)

    # 학습 데이터 클래스 카운트 가중치(선택)
    if class_counts:
        # 키가 '0','1' 문자열일 수도 있어서 양쪽 접근
        def _get_cc(i):
            return class_counts.get(i, class_counts.get(str(i), 0))
        total_class = float(sum(float(v) for v in class_counts.values())) or 1.0
        class_weights = np.array([
            np.exp(beta * (1.0 - float(_get_cc(i)) / total_class))
            for i in range(num_classes)
        ], dtype=np.float64)
    else:
        class_weights = np.ones(num_classes, dtype=np.float64)

    class_weights = np.clip(class_weights, 0.85, 1.15)

    combined = np.clip(recent_weights * class_weights, 0.85, 1.15)
    adjusted = p * combined
    s = adjusted.sum()
    if s <= 0 or not np.isfinite(s):
        return p
    return adjusted / s

# (엔트리 포인트용)
if __name__ == "__main__":
    run()
