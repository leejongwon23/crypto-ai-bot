# === predict_trigger.py (MEM-SAFE FINAL++++ — group-finished only, hard gate enforced, stale-lock safe) ===
import os
import time
import traceback
import datetime
from collections import Counter
import glob

import numpy as np
import pandas as pd
import pytz

# ──────────────────────────────────────────────────────────────
# 데이터 소스 (패키지/루트 폴백)
# ──────────────────────────────────────────────────────────────
try:
    from data.utils import get_ALL_SYMBOLS, get_kline_by_strategy
except Exception:
    try:
        from utils import get_ALL_SYMBOLS, get_kline_by_strategy  # 루트 폴백
    except Exception as _e:
        def get_ALL_SYMBOLS():
            print(f"[경고] get_ALL_SYMBOLS 임포트 실패: {_e}")
            return []
        def get_kline_by_strategy(symbol, strategy):
            print(f"[경고] get_kline_by_strategy 임포트 실패: {symbol}-{strategy} / {_e}")
            return None

# 로깅 보장
from logger import log_audit_prediction as log_audit, ensure_prediction_log_exists

# 전역 리셋/정리 락
try:
    import safe_cleanup
    _LOCK_PATH = getattr(safe_cleanup, "LOCK_PATH", "/persistent/locks/train_or_predict.lock")
except Exception:
    _LOCK_PATH = "/persistent/locks/train_or_predict.lock"

# 예측 게이트/락 경로
PREDICT_BLOCK    = "/persistent/predict.block"
PREDICT_RUN_LOCK = "/persistent/run/predict_running.lock"
GROUP_TRAIN_LOCK = "/persistent/run/group_training.lock"

# 모델 경로
MODEL_DIR   = "/persistent/models"
_KNOWN_EXTS = (".pt", ".ptz", ".safetensors")

# 전략 집합
STRATEGIES  = ["단기", "중기", "장기"]

def _has_model_for(symbol: str, strategy: str) -> bool:
    try:
        for e in _KNOWN_EXTS:
            if glob.glob(os.path.join(MODEL_DIR, f"{symbol}_{strategy}_*{e}")):
                return True
        d = os.path.join(MODEL_DIR, symbol, strategy)
        if os.path.isdir(d):
            for e in _KNOWN_EXTS:
                if glob.glob(os.path.join(d, f"*{e}")):
                    return True
    except Exception:
        pass
    return False

# (옵션) 레짐/캘리브레이션 정보
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

# (옵션) 예측 호출 래퍼
_safe_predict_with_timeout = None
_safe_predict_sync = None
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

# (옵션) 예측 게이트 상태 API
try:
    from predict import is_predict_gate_open as __is_open
except Exception:
    __is_open = None

# 그룹 오더 매니저
_GOM = None
try:
    from group_order import GroupOrderManager as _GOM
except Exception:
    try:
        from data.group_order import GroupOrderManager as _GOM
    except Exception:
        _GOM = None

def _get_current_group_symbols():
    if _GOM is None:
        return None
    try:
        gom = _GOM()
        if hasattr(gom, "get_current_group_symbols"):
            syms = gom.get_current_group_symbols()
        elif hasattr(gom, "current_group_index") and hasattr(gom, "get_group_symbols"):
            syms = gom.get_group_symbols(gom.current_group_index())
        else:
            return None
        if not syms:
            return None
        return list(dict.fromkeys(syms))
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────
TRIGGER_COOLDOWN = {"단기": 3600, "중기": 10800, "장기": 21600}
MAX_LOOKBACK = int(os.getenv("TRIGGER_MAX_LOOKBACK", "180"))
RECENT_DAYS_FOR_FREQ = max(1, int(os.getenv("TRIGGER_FREQ_DAYS", "3")))
CSV_CHUNKSIZE = max(10000, int(os.getenv("TRIGGER_CSV_CHUNKSIZE", "50000")))
TRIGGER_MAX_PER_RUN = max(1, int(os.getenv("TRIGGER_MAX_PER_RUN", "999")))
PREDICT_TIMEOUT_SEC = float(os.getenv("PREDICT_TIMEOUT_SEC", "30"))
PREDICT_LOCK_STALE_TRIGGER_SEC = int(os.getenv("PREDICT_LOCK_STALE_TRIGGER_SEC", "600"))

# 그룹 완료 모드
REQUIRE_GROUP_COMPLETE = int(os.getenv("REQUIRE_GROUP_COMPLETE", "0"))

last_trigger_time = {}
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# ──────────────────────────────────────────────────────────────
# 게이트/락 관리
# ──────────────────────────────────────────────────────────────
def _gate_closed() -> bool:
    try:
        if os.path.exists(GROUP_TRAIN_LOCK):
            return True
        if os.path.exists(PREDICT_BLOCK):
            return True
    except Exception:
        pass
    return False

def _predict_busy() -> bool:
    try:
        return os.path.exists(PREDICT_RUN_LOCK)
    except Exception:
        return False

def _is_stale_lock(path: str, ttl_sec: int) -> bool:
    try:
        if not os.path.exists(path):
            return False
        mtime = os.path.getmtime(path)
        return (time.time() - float(mtime)) > max(30, int(ttl_sec))
    except Exception:
        return False

def _clear_stale_predict_lock(ttl_sec: int):
    try:
        if _is_stale_lock(PREDICT_RUN_LOCK, ttl_sec):
            os.remove(PREDICT_RUN_LOCK)
            print(f"[LOCK] stale predict lock removed (> {ttl_sec}s)")
    except Exception as e:
        print(f"[LOCK] stale cleanup error: {e}")

# ──────────────────────────────────────────────────────────────
# 그룹 완성 검사
# ──────────────────────────────────────────────────────────────
def _missing_pairs(symbols):
    miss = []
    for sym in symbols:
        for st in STRATEGIES:
            if not _has_model_for(sym, st):
                miss.append((sym, st))
    return miss

def _available_pairs(symbols):
    for sym in symbols:
        for st in STRATEGIES:
            if _has_model_for(sym, st):
                yield sym, st

def _is_group_complete_for_all_strategies(symbols) -> bool:
    return len(_missing_pairs(symbols)) == 0

# ──────────────────────────────────────────────────────────────
# 전조 조건
# ──────────────────────────────────────────────────────────────
def _has_cols(df: pd.DataFrame, cols) -> bool:
    return isinstance(df, pd.DataFrame) and set(cols).issubset(set(df.columns))

def check_pre_burst_conditions(df, strategy):
    try:
        if df is None or len(df) < 10 or not _has_cols(df, ["close"]):
            return True
        if MAX_LOOKBACK > 0 and len(df) > MAX_LOOKBACK:
            df = df.tail(MAX_LOOKBACK)

        if "volume" in df.columns and len(df) >= 3:
            vol_increasing = df["volume"].iloc[-3] < df["volume"].iloc[-2] < df["volume"].iloc[-1]
        else:
            vol_increasing = False

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
    return _has_model_for(symbol, strategy)

# ──────────────────────────────────────────────────────────────
# 트리거 실행 루프
# ──────────────────────────────────────────────────────────────
def run():
    if _LOCK_PATH and os.path.exists(_LOCK_PATH):
        print(f"[트리거] 전역 락 감지({_LOCK_PATH}) → 전체 스킵 @ {now_kst().isoformat()}")
        return

    _clear_stale_predict_lock(PREDICT_LOCK_STALE_TRIGGER_SEC)

    if _gate_closed():
        print(f"[트리거] 그룹 학습 중/비상차단 → 스킵 @ {now_kst().isoformat()}")
        return

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

    try:
        all_symbols = list(dict.fromkeys(get_ALL_SYMBOLS()))
    except Exception as e:
        print(f"[경고] 심볼 로드 실패: {e}")
        all_symbols = []

    group_syms = _get_current_group_symbols()
    if isinstance(group_syms, (list, tuple)) and len(group_syms) > 0:
        symset = set(group_syms)
        symbols = [s for s in all_symbols if s in symset]
        print(f"[그룹제한] 현재 그룹 심볼 {len(symbols)}/{len(all_symbols)}개 대상으로 실행")

        if REQUIRE_GROUP_COMPLETE and not _is_group_complete_for_all_strategies(symbols):
            miss = _missing_pairs(symbols)
            print(f"[차단] 그룹 미완료(누락 {len(miss)}) {miss} → 예측 실행 안 함")
            return
    else:
        symbols = all_symbols

    print(f"[트리거 시작] {now_kst().isoformat()} / 대상 심볼 {len(symbols)}개")

    triggered = 0
    target_pairs = list(_available_pairs(symbols))

    for symbol, strategy in target_pairs:
        if triggered >= TRIGGER_MAX_PER_RUN:
            print(f"[트리거] 이번 루프 최대 실행 수({TRIGGER_MAX_PER_RUN}) 도달 → 조기 종료")
            print(f"🔁 이번 트리거 루프에서 예측 실행된 개수: {triggered}")
            return

        _clear_stale_predict_lock(PREDICT_LOCK_STALE_TRIGGER_SEC)
        if _LOCK_PATH and os.path.exists(_LOCK_PATH):
            print(f"[트리거] 실행 중 전역 락 감지 → 중단")
            print(f"🔁 이번 트리거 루프에서 예측 실행된 개수: {triggered}")
            return
        if _gate_closed() or _predict_busy():
            print(f"[트리거] 게이트 닫힘/예측 중 → 중단")
            print(f"🔁 이번 트리거 루프에서 예측 실행된 개수: {triggered}")
            return

        try:
            key = f"{symbol}_{strategy}"
            now = time.time()
            cooldown = TRIGGER_COOLDOWN.get(strategy, 3600)
            if now - last_trigger_time.get(key, 0) < cooldown:
                continue

            if not check_model_quality(symbol, strategy):
                continue

            df = get_kline_by_strategy(symbol, strategy)
            if df is None or len(df) < 60 or not _has_cols(df, ["close"]):
                continue

            if not check_pre_burst_conditions(df, strategy):
                continue

            try:
                regime = detect_regime(symbol, strategy, now=now_kst())
                calib_ver = get_calibration_version()
                log_audit(symbol, strategy, "프리로드", f"regime={regime}, calib_ver={calib_ver}")
            except Exception as preload_e:
                print(f"[프리로드 경고] {symbol}-{strategy}: {preload_e}")

            print(f"[✅ 트리거 포착] {symbol} - {strategy} → 예측 실행")

            try:
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
                elif _safe_predict_sync:
                    _safe_predict_sync(
                        predict_fn=_predict,
                        symbol=symbol,
                        strategy=strategy,
                        source="변동성",
                        model_type=None,
                    )
                else:
                    _predict(symbol, strategy, source="변동성")
                last_trigger_time[key] = now
                log_audit(symbol, strategy, "트리거예측", "조건 만족으로 실행")
                triggered += 1
            except Exception as inner:
                print(f"[❌ 예측 실행 실패] {symbol}-{strategy}: {inner}")
                log_audit(symbol, strategy, "트리거예측오류", f"예측실행실패: {inner}")
        except Exception as e:
            print(f"[트리거 오류] {symbol} {strategy}: {e}")
            log_audit(symbol, strategy or "알수없음", "트리거오류", str(e))

    print(f"🔁 이번 트리거 루프에서 예측 실행된 개수: {triggered}")

# ──────────────────────────────────────────────────────────────
# 최근 클래스 빈도
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
# 확률 보정
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
            np.exp(-alpha * (float(recent_freq.get(i,
