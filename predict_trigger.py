# === predict_trigger.py (MEM-SAFE FINAL+++ — gate/lock aware, stale lock cleanup, timeout-safe, freq & diversity) ===
import os
import time
import traceback
import datetime
from collections import Counter

import numpy as np
import pandas as pd
import pytz

# ✅ 단일 소스 심볼/데이터
from data.utils import get_ALL_SYMBOLS, get_kline_by_strategy

# ✅ 로그 보장
from logger import log_audit_prediction as log_audit, ensure_prediction_log_exists

# ✅ 전역 락(RESET/초기화 중) 감지 → 전체 트리거 스킵
try:
    import safe_cleanup
    _LOCK_PATH = getattr(safe_cleanup, "LOCK_PATH", "/persistent/locks/train_or_predict.lock")
except Exception:
    _LOCK_PATH = "/persistent/locks/train_or_predict.lock"

# ✅ 그룹예측 게이트/락 (predict.py와 합의된 경로)
PREDICT_BLOCK = "/persistent/predict.block"
PREDICT_RUN_LOCK = "/persistent/run/predict_running.lock"

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

# ▷ (옵션) 예측 실행 호출 래퍼 — train.py에서 제공(있으면 사용)
_safe_predict_with_timeout = None   # (선호) 타임아웃 지원 버전
_safe_predict_sync = None           # (대안) 동기 버전
try:
    from train import _safe_predict_with_timeout as __t_safe_to
    _safe_predict_with_timeout = __t_safe_to
except Exception:
    pass
try:
    from train import _safe_predict_sync as __t_safe_sync
    _safe_predict_sync = __t_safe_sync
except Exception:
    pass

# ▷ (옵션) 게이트 상태 확인 API (predict.py)
_is_gate_open = None
try:
    from predict import is_predict_gate_open as __is_open
    _is_gate_open = __is_open
except Exception:
    _is_gate_open = None

# ===== 설정(환경변수로 조절 가능) =====
TRIGGER_COOLDOWN = {"단기": 3600, "중기": 10800, "장기": 21600}
MODEL_TYPES = ["lstm", "cnn_lstm", "transformer"]
MAX_LOOKBACK = int(os.getenv("TRIGGER_MAX_LOOKBACK", "180"))   # 전조 계산시 최근 N행만 사용
RECENT_DAYS_FOR_FREQ = max(1, int(os.getenv("TRIGGER_FREQ_DAYS", "3")))
CSV_CHUNKSIZE = max(10000, int(os.getenv("TRIGGER_CSV_CHUNKSIZE", "50000")))
TRIGGER_MAX_PER_RUN = max(1, int(os.getenv("TRIGGER_MAX_PER_RUN", "999")))  # 1회 루프에서 최대 실행 수
PREDICT_TIMEOUT_SEC = float(os.getenv("PREDICT_TIMEOUT_SEC", "30"))         # _safe_predict_with_timeout 없을 때는 미사용

# 🔧 stale lock(고아 락) 처리 임계
PREDICT_LOCK_STALE_TRIGGER_SEC = int(os.getenv("PREDICT_LOCK_STALE_TRIGGER_SEC", "120"))

last_trigger_time = {}
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# ──────────────────────────────────────────────────────────────
# 유틸: 게이트/락, stale lock 정리
# ──────────────────────────────────────────────────────────────
def _gate_closed() -> bool:
    """그룹 예측 중에는 조용히 스킵(실제 예측 호출 자체를 피함)."""
    try:
        if os.path.exists(PREDICT_BLOCK):
            return True
        if _is_gate_open is not None and (not _is_gate_open()):
            return True
    except Exception:
        pass
    return False

def _predict_busy() -> bool:
    """동시에 predict가 이미 돌고 있으면 조용히 스킵."""
    try:
        return os.path.exists(PREDICT_RUN_LOCK)
    except Exception:
        return False

def _is_stale_lock(path: str, ttl_sec: int) -> bool:
    try:
        if not os.path.exists(path): return False
        mtime = os.path.getmtime(path)
        return (time.time() - float(mtime)) > max(30, int(ttl_sec))
    except Exception:
        return False

def _clear_stale_predict_lock(ttl_sec: int):
    """오래된 고아 락 자동 제거(예: 이전 예측 중 비정상 종료)."""
    try:
        if _is_stale_lock(PREDICT_RUN_LOCK, ttl_sec):
            os.remove(PREDICT_RUN_LOCK)
            print(f"[LOCK] stale predict lock removed (> {ttl_sec}s)")
    except Exception as e:
        print(f"[LOCK] stale cleanup error: {e}")

# ──────────────────────────────────────────────────────────────
# 전조 조건(메모리/연산 예산 보호 포함)
# ──────────────────────────────────────────────────────────────
def _has_cols(df: pd.DataFrame, cols) -> bool:
    return isinstance(df, pd.DataFrame) and set(cols).issubset(set(df.columns))

def check_pre_burst_conditions(df, strategy):
    try:
        if df is None or len(df) < 10 or not _has_cols(df, ["close"]):
            print("[경고] 데이터 너무 적음/컬럼부족 → fallback 조건 평가")
            return True

        # 메모리/연산량 절약: 최근 구간만 사용
        if MAX_LOOKBACK > 0 and len(df) > MAX_LOOKBACK:
            df = df.tail(MAX_LOOKBACK)

        # 방어: volume 없으면 단조 증가 체크 건너뜀
        if "volume" in df.columns and len(df) >= 3:
            vol_increasing = df["volume"].iloc[-3] < df["volume"].iloc[-2] < df["volume"].iloc[-1]
        else:
            vol_increasing = False

        # 가격 안정/압축
        price_range = df["close"].iloc[-min(len(df), 6):]
        stable_price = (price_range.max() - price_range.min()) / (price_range.mean() + 1e-12) < 0.005

        ema_5  = df["close"].ewm(span=5).mean().iloc[-1]  if len(df) >= 5  else df["close"].mean()
        ema_15 = df["close"].ewm(span=15).mean().iloc[-1] if len(df) >= 15 else df["close"].mean()
        ema_60 = df["close"].ewm(span=60).mean().iloc[-1] if len(df) >= 60 else df["close"].mean()
        ema_pack = max(ema_5, ema_15, ema_60) - min(ema_5, ema_15, ema_60)
        ema_compressed = ema_pack / (float(df["close"].iloc[-1]) + 1e-12) < 0.003

        if len(df) >= 20:
            bb_std = df["close"].rolling(window=20).std()
            expanding_band = (bb_std.iloc[-2] < bb_std.iloc[-1]) and (bb_std.iloc[-1] > 0.002)
        else:
            expanding_band = True  # 짧은 시계열은 관대한 기준

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
    # TODO: 필요시 메타/성능 기준 추가
    return True

# ──────────────────────────────────────────────────────────────
# 트리거 실행 루프(락/쿨다운/최대 실행 수/타임아웃 지원)
# ──────────────────────────────────────────────────────────────
def run():
    # 전역 락이면 전체 스킵
    if _LOCK_PATH and os.path.exists(_LOCK_PATH):
        print(f"[트리거] 전역 락 감지({_LOCK_PATH}) → 전체 스킵 @ {now_kst().isoformat()}")
        return

    # 예측 고아 락 정리(있다면)
    _clear_stale_predict_lock(PREDICT_LOCK_STALE_TRIGGER_SEC)

    # 게이트 닫힘이면 전체 스킵(그룹 학습/예측 중일 수 있음)
    if _gate_closed():
        print(f"[트리거] 게이트 닫힘(그룹 예측 진행 중) → 스킵 @ {now_kst().isoformat()}")
        return

    # 이미 예측 중이면 스킵(중복 실행 방지)
    if _predict_busy():
        print(f"[트리거] 예측 실행 중(lock) → 스킵 @ {now_kst().isoformat()}")
        return

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

    # 심볼 목록 방어: 중복 제거 + 정렬(안정적 순회)
    try:
        symbols = list(dict.fromkeys(get_ALL_SYMBOLS()))
    except Exception as e:
        print(f"[경고] 심볼 로드 실패: {e}")
        symbols = []

    for symbol in symbols:
        for strategy in ["단기", "중기", "장기"]:
            # 최대 실행 수 초과 시 즉시 종료(스케줄 다음 턴으로 넘김)
            if triggered >= TRIGGER_MAX_PER_RUN:
                print(f"[트리거] 이번 루프 최대 실행 수({TRIGGER_MAX_PER_RUN}) 도달 → 조기 종료")
                print(f"🔁 이번 트리거 루프에서 예측 실행된 개수: {triggered}")
                return

            # 실행 중간에도 락/게이트 상태 변하면 조용히 종료
            if _LOCK_PATH and os.path.exists(_LOCK_PATH):
                print(f"[트리거] 실행 중 전역 락 감지 → 중단")
                print(f"🔁 이번 트리거 루프에서 예측 실행된 개수: {triggered}")
                return
            _clear_stale_predict_lock(PREDICT_LOCK_STALE_TRIGGER_SEC)
            if _gate_closed() or _predict_busy():
                print(f"[트리거] 게이트 닫힘/예측 중 → 스킵")
                return

            try:
                key = f"{symbol}_{strategy}"
                now = time.time()
                cooldown = TRIGGER_COOLDOWN.get(strategy, 3600)

                if now - last_trigger_time.get(key, 0) < cooldown:
                    # 너무 시끄럽지 않게 간단 출력
                    continue

                df = get_kline_by_strategy(symbol, strategy)
                if df is None or len(df) < 60 or not _has_cols(df, ["close"]):
                    # 데이터 부족/컬럼 부족
                    continue

                if not check_model_quality(symbol, strategy):
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
                        # 1순위: 타임아웃 지원 호출
                        if _safe_predict_with_timeout:
                            ok = _safe_predict_with_timeout(
                                predict_fn=_predict,
                                symbol=symbol,
                                strategy=strategy,
                                source="변동성",
                                model_type=None,
                                timeout=PREDICT_TIMEOUT_SEC,
                            )
                            if not ok:
                                raise RuntimeError("predict timeout/failed")
                        # 2순위: 동기 호출 래퍼(타임아웃 없음)
                        elif _safe_predict_sync:
                            _safe_predict_sync(
                                predict_fn=_predict,
                                symbol=symbol,
                                strategy=strategy,
                                source="변동성",
                                model_type=None,
                            )
                        else:
                            # 3순위: 직접 호출(타임아웃 미지원, predict.py 내부에서 gate/lock/heartbeat 처리)
                            _predict(symbol, strategy, source="변동성")

                        last_trigger_time[key] = now
                        log_audit(symbol, strategy, "트리거예측", "조건 만족으로 실행")
                        triggered += 1
                    except Exception as inner:
                        print(f"[❌ 예측 실행 실패] {symbol}-{strategy}: {inner}")
                        log_audit(symbol, strategy, "트리거예측오류", f"예측실행실패: {inner}")
                # else: 조건 미충족은 조용히 넘어감

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
            on_bad_lines="skip",
        ):
            if "predicted_class" not in chunk.columns or "timestamp" not in chunk.columns:
                continue

            if strategy and "strategy" in chunk.columns:
                chunk = chunk[chunk["strategy"] == strategy]

            ts = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
            try:
                ts = ts.dt.tz_convert("Asia/Seoul")
            except Exception:
                ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")

            mask = ts >= cutoff
            if not mask.any():
                continue

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
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 2:
        p = p[0]
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.clip(p, 0.0, None)
    s0 = p.sum()
    if s0 <= 0:
        return np.ones_like(p) / max(1, len(p))
    p = p / s0

    num_classes = len(p)

    total_recent = float(sum(recent_freq.values()))
    if total_recent <= 0:
        recent_weights = np.ones(num_classes, dtype=np.float64)
    else:
        recent_weights = np.array([
            np.exp(-alpha * (float(recent_freq.get(i, 0)) / total_recent))
            for i in range(num_classes)
        ], dtype=np.float64)
        recent_weights = np.clip(recent_weights, 0.85, 1.15)

    if class_counts:
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
