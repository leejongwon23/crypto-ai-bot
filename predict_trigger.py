# === predict_trigger.py (FINAL — lock-aware, retry-on-unlock, stale-safe, log-throttled) ===
import os
import time
import traceback
import datetime
from collections import Counter, defaultdict
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

# 예측 게이트/락 경로(파일 폴백)
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

# (옵션) 중앙 락 유틸 사용: 있으면 우선 사용
_lock_api = {"is_locked": None, "clear_stale": None, "wait_until_free": None, "ttl": None}
try:
    from predict_lock import is_predict_running as _is_locked
    from predict_lock import clear_stale_predict_lock as _clear_stale
    from predict_lock import wait_until_free as _wait_until_free
    from predict_lock import PREDICT_LOCK_TTL as _LOCK_TTL
    _lock_api.update(is_locked=_is_locked, clear_stale=_clear_stale,
                     wait_until_free=_wait_until_free, ttl=int(_LOCK_TTL))
except Exception:
    pass  # 파일기반 폴백 사용

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

# ✅ 재시도/대기/쓰로틀 설정
RETRY_AFTER_TRAIN_MAX_WAIT_SEC   = int(os.getenv("RETRY_AFTER_TRAIN_MAX_WAIT_SEC", "900"))
RETRY_AFTER_TRAIN_SLEEP_SEC      = float(os.getenv("RETRY_AFTER_TRAIN_SLEEP_SEC", "1.0"))
STARTUP_WAIT_FOR_GATE_OPEN_SEC   = int(os.getenv("STARTUP_WAIT_FOR_GATE_OPEN_SEC", "600"))
PAIR_WAIT_FOR_GATE_OPEN_SEC      = int(os.getenv("PAIR_WAIT_FOR_GATE_OPEN_SEC", "120"))
RETRY_ON_TIMEOUT                 = int(os.getenv("RETRY_ON_TIMEOUT", "1")) == 1
TIMEOUT_RETRY_ONCE_EXTRA_SEC     = float(os.getenv("TIMEOUT_RETRY_ONCE_EXTRA_SEC", "20"))
# 쓰로틀: 바쁜상태/타임아웃 로그 최소 간격
THROTTLE_BUSY_LOG_SEC            = int(os.getenv("THROTTLE_BUSY_LOG_SEC", "15"))
# 페어별 백오프(타임아웃/락실패 반복 방지)
PAIR_BACKOFF_BASE_SEC            = int(os.getenv("PAIR_BACKOFF_BASE_SEC", "60"))
PAIR_BACKOFF_MAX_SEC             = int(os.getenv("PAIR_BACKOFF_MAX_SEC", "600"))

# 그룹 완료 모드
REQUIRE_GROUP_COMPLETE = int(os.getenv("REQUIRE_GROUP_COMPLETE", "0"))

last_trigger_time = {}
_last_busy_log_at = 0.0
_pair_backoff_until = defaultdict(float)   # key -> unix ts
_pair_backoff_step  = defaultdict(int)     # key -> step

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# ──────────────────────────────────────────────────────────────
# 게이트/락 관리 (파일 폴백 포함)
# ──────────────────────────────────────────────────────────────
def _gate_closed() -> bool:
    try:
        if os.path.exists(GROUP_TRAIN_LOCK):
            return True
        if os.path.exists(PREDICT_BLOCK):
            return True
        if __is_open is not None:
            # predict.py 게이트 API가 있으면 신뢰
            return (not bool(__is_open()))
    except Exception:
        pass
    return False

def _predict_busy() -> bool:
    # 중앙 락 API가 있으면 우선
    if callable(_lock_api["is_locked"]):
        try:
            return bool(_lock_api["is_locked"]())
        except Exception:
            pass
    # 파일 폴백
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
    # 중앙 락 API가 있으면 우선
    if callable(_lock_api["clear_stale"]):
        try:
            _lock_api["clear_stale"]()
            return
        except Exception:
            pass
    try:
        if _is_stale_lock(PREDICT_RUN_LOCK, ttl_sec):
            os.remove(PREDICT_RUN_LOCK)
            print(f"[LOCK] stale predict lock removed (> {ttl_sec}s)")
    except Exception as e:
        print(f"[LOCK] stale cleanup error: {e}")

def _wait_for_gate_open(max_wait_sec: int) -> bool:
    """게이트/락이 열릴 때까지 최대 max_wait_sec 동안 대기."""
    start = time.time()
    while time.time() - start < max_wait_sec:
        # 1) stale 정리(안전)
        _clear_stale_predict_lock(PREDICT_LOCK_STALE_TRIGGER_SEC)

        # 2) 전역 유지보수 락이면 즉시 포기
        if _LOCK_PATH and os.path.exists(_LOCK_PATH):
            return False

        # 3) 게이트가 열려 있고, 예측 락도 비어있으면 통과
        if (not _gate_closed()) and (not _predict_busy()):
            return True

        # 4) 중앙 wait API 있으면 활용(짧게 대기)
        if callable(_lock_api["wait_until_free"]):
            try:
                if _lock_api["wait_until_free"](max_wait_sec=1):
                    # 락은 비었으나 게이트가 닫혀있을 수 있음 — 루프 재평가
                    pass
            except Exception:
                pass
        time.sleep(max(0.05, RETRY_AFTER_TRAIN_SLEEP_SEC))
    return False

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
# 내부: 예측 실행(타임아웃/재시도 포함)
# ──────────────────────────────────────────────────────────────
def _invoke_predict(_predict, symbol, strategy, source, timeout_sec: float) -> bool:
    """timeout 지원 래퍼 (train 제공 함수가 있으면 사용)"""
    if _safe_predict_with_timeout:
        ok = _safe_predict_with_timeout(
            predict_fn=_predict,
            symbol=symbol,
            strategy=strategy,
            source=source,
            model_type=None,
            timeout=timeout_sec,
        )
        return bool(ok)
    elif _safe_predict_sync:
        _safe_predict_sync(
            predict_fn=_predict,
            symbol=symbol,
            strategy=strategy,
            source=source,
            model_type=None,
        )
        return True
    else:
        _predict(symbol, strategy, source=source)
        return True

def _retry_after_training(_predict, symbol, strategy, first_err: Exception | str = None) -> bool:
    """훈련락/게이트 해제까지 기다렸다가 1회 재시도"""
    why = f"timeout/lock; first_err={first_err}" if first_err else "timeout/lock"
    log_audit(symbol, strategy, "트리거재시도대기", why)
    ok = _wait_for_gate_open(RETRY_AFTER_TRAIN_MAX_WAIT_SEC)
    if not ok:
        log_audit(symbol, strategy, "트리거재시도포기", "게이트 미오픈(대기초과)")
        return False
    try:
        ok2 = _invoke_predict(_predict, symbol, strategy, "변동성(재시도)", max(PREDICT_TIMEOUT_SEC, TIMEOUT_RETRY_ONCE_EXTRA_SEC))
        if ok2:
            log_audit(symbol, strategy, "트리거예측(재시도성공)", "훈련락 해제 후 성공")
        else:
            log_audit(symbol, strategy, "트리거예측(재시도실패)", "재시도 실패")
        return bool(ok2)
    except Exception as e:
        log_audit(symbol, strategy, "트리거예측(재시도예외)", f"{e}")
        return False

# ──────────────────────────────────────────────────────────────
# 트리거 실행 루프
# ──────────────────────────────────────────────────────────────
def run():
    global _last_busy_log_at

    # 전역 강제 락: 즉시 스킵
    if _LOCK_PATH and os.path.exists(_LOCK_PATH):
        print(f"[트리거] 전역 락 감지({_LOCK_PATH}) → 전체 스킵 @ {now_kst().isoformat()}")
        return

    _clear_stale_predict_lock(PREDICT_LOCK_STALE_TRIGGER_SEC)

    # 시작 시 게이트 닫힘/바쁨이면 일정 시간 대기
    if _gate_closed() or _predict_busy():
        print(f"[트리거] 시작 시 게이트 닫힘/예측중 → 최대 {STARTUP_WAIT_FOR_GATE_OPEN_SEC}s 대기")
        opened = _wait_for_gate_open(STARTUP_WAIT_FOR_GATE_OPEN_SEC)
        if not opened:
            print(f"[트리거] 게이트 미오픈(대기초과) → 스킵 @ {now_kst().isoformat()}")
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

        # 페어별 백오프 적용
        key = f"{symbol}_{strategy}"
        nowu = time.time()
        if nowu < _pair_backoff_until[key]:
            continue

        _clear_stale_predict_lock(PREDICT_LOCK_STALE_TRIGGER_SEC)

        # 실행 중 전역 락/게이트 닫힘 → 대기
        if _LOCK_PATH and os.path.exists(_LOCK_PATH):
            print(f"[트리거] 실행 중 전역 락 감지 → 중단")
            print(f"🔁 이번 트리거 루프에서 예측 실행된 개수: {triggered}")
            return

        if _gate_closed() or _predict_busy():
            # 로그 쓰로틀
            if (nowu - _last_busy_log_at) >= THROTTLE_BUSY_LOG_SEC:
                print(f"[트리거] 게이트 닫힘/예측중 → 최대 {PAIR_WAIT_FOR_GATE_OPEN_SEC}s 대기 후 재시도")
                _last_busy_log_at = nowu
            opened = _wait_for_gate_open(PAIR_WAIT_FOR_GATE_OPEN_SEC)
            if not opened:
                if (nowu - _last_busy_log_at) >= THROTTLE_BUSY_LOG_SEC:
                    print(f"[트리거] 게이트 미오픈(대기초과) → 중단")
                    _last_busy_log_at = nowu
                print(f"🔁 이번 트리거 루프에서 예측 실행된 개수: {triggered}")
                return

        try:
            nowt = time.time()
            cooldown = TRIGGER_COOLDOWN.get(strategy, 3600)
            if nowt - last_trigger_time.get(key, 0) < cooldown:
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
                ok = _invoke_predict(_predict, symbol, strategy, "변동성", PREDICT_TIMEOUT_SEC)
                if not ok and RETRY_ON_TIMEOUT:
                    ok = _retry_after_training(_predict, symbol, strategy, first_err="timeout/failed")

                if ok:
                    last_trigger_time[key] = nowt
                    # 성공 시 백오프 해제/초기화
                    _pair_backoff_until.pop(key, None)
                    _pair_backoff_step.pop(key, None)
                    log_audit(symbol, strategy, "트리거예측", "조건 만족으로 실행")
                    triggered += 1
                else:
                    # 실패 시 페어 백오프(지수 증가, 상한 있음)
                    step = _pair_backoff_step[key] = min(_pair_backoff_step[key] + 1, 8)
                    wait_sec = min(PAIR_BACKOFF_BASE_SEC * (2 ** (step - 1)), PAIR_BACKOFF_MAX_SEC)
                    _pair_backoff_until[key] = time.time() + wait_sec
                    raise RuntimeError(f"predict timeout/failed (backoff {wait_sec}s)")
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
    """
    probs           : (C,) or (1,C) 확률 벡터
    recent_freq     : 최근 예측된 클래스 빈도 Counter
    class_counts    : (선택) 학습시 클래스 샘플 수 {cls: count}
    alpha           : 최근 과다선택된 클래스 패널티 강도
    beta            : 데이터 희소클래스 보정 강도
    반환            : 정규화된 (C,) 벡터
    """
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

    # 최근 빈도 기반 가중치 (많이 나왔던 클래스 패널티)
    total_recent = float(sum(recent_freq.values()))
    if total_recent <= 0:
        recent_weights = np.ones(num_classes, dtype=np.float64)
    else:
        recent_weights = np.array([
            np.exp(-alpha * (float(recent_freq.get(i, 0)) / total_recent))
            for i in range(num_classes)
        ], dtype=np.float64)
        recent_weights = np.clip(recent_weights, 0.5, 1.5)

    # (선택) 클래스 데이터 수 기반 희소성 보정
    if class_counts and isinstance(class_counts, dict):
        counts = np.array([float(class_counts.get(i, 0.0)) for i in range(num_classes)], dtype=np.float64)
        counts = np.where(np.isfinite(counts), counts, 0.0)
        inv_sqrt = 1.0 / np.sqrt(counts + 1e-6)
        inv_sqrt = inv_sqrt / (inv_sqrt.mean() + 1e-12)
        rarity_weights = (1.0 - beta) + beta * inv_sqrt
    else:
        rarity_weights = np.ones(num_classes, dtype=np.float64)

    w = recent_weights * rarity_weights
    w = np.where(np.isfinite(w), w, 1.0)
    w = np.clip(w, 1e-6, None)

    p_adj = p * w
    s = p_adj.sum()
    if s <= 0:
        return np.ones_like(p) / max(1, len(p))
    return (p_adj / s).astype(np.float64)
