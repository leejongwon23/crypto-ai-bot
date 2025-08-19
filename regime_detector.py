# === regime_detector.py (FINAL) ===
"""
현재 시장 상태(레짐)를 가볍게 태깅해서 모델/전략 선택에 쓰기 위한 모듈.
- 외부 의존성: pandas, numpy (프로젝트 기본)
- 설정: config.get_REGIME() (enabled=False면 항상 '중간/중립'으로 리턴)
- 데이터: data.utils.get_kline_by_strategy()
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd

# 프로젝트 내부
from data.utils import get_kline_by_strategy
try:
    from config import get_REGIME
except Exception:
    # 안전 기본값 (설정 파일이 없거나 로드 실패시)
    def get_REGIME():
        return {
            "enabled": False,
            "atr_window": 14,
            "rsi_window": 14,
            "trend_window": 50,
            "vol_high_pct": 0.9,
            "vol_low_pct": 0.5
        }

# 간단 캐시(5분)
_CACHE: dict[tuple[str, str], tuple[float, dict]] = {}
_CACHE_TTL = 300.0

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (NaN → 0으로 채움)."""
    hl = (high - low).abs()
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr.fillna(0.0)

def _trend_slope(close: pd.Series, window: int = 50) -> pd.Series:
    """단순 이동평균의 기울기(1차 차분)."""
    ma = close.rolling(window=window, min_periods=1).mean()
    slope = ma.diff().fillna(0.0)
    return slope

def detect_regime(symbol: str, strategy: str, now: pd.Timestamp | None = None) -> dict:
    """
    현재 레짐을 계산해 dict로 반환.
    return 예시:
      {
        "enabled": True,
        "vol_regime": 0|1|2,   # 0=낮음, 1=중간, 2=높음
        "trend_regime": 0|1|2, # 0=하락, 1=중립, 2=상승
        "regime_tag": int,     # vol*3 + trend (0~8)
        "stats": { "atr_q_low":..., "atr_q_high":..., "atr_last":..., "slope_last":... },
      }
    """
    cfg = get_REGIME()
    if not cfg.get("enabled", False):
        # 꺼져 있으면 항상 중립 반환(기존 동작과 동일)
        return {
            "enabled": False,
            "vol_regime": 1,
            "trend_regime": 1,
            "regime_tag": 4,  # 1*3 + 1
            "stats": {}
        }

    # 캐시 확인
    key = (symbol, strategy)
    t = time.time()
    hit = _CACHE.get(key)
    if hit and (t - hit[0] < _CACHE_TTL):
        return hit[1]

    # 데이터 확보 (가벼운 호출)
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or df.empty or "close" not in df:
        # 데이터 없으면 중립
        out = {
            "enabled": True,
            "vol_regime": 1,
            "trend_regime": 1,
            "regime_tag": 4,
            "stats": {"reason": "no_data"}
        }
        _CACHE[key] = (t, out)
        return out

    # 최근 구간만 사용(불필요한 메모리/연산 방지)
    df = df.tail(600).copy()
    for c in ["high", "low", "close"]:
        if c not in df:
            df[c] = df["close"]
    df = df.dropna(subset=["high", "low", "close"])
    if df.empty:
        out = {
            "enabled": True,
            "vol_regime": 1,
            "trend_regime": 1,
            "regime_tag": 4,
            "stats": {"reason": "no_valid_rows"}
        }
        _CACHE[key] = (t, out)
        return out

    # 파라미터
    atr_win   = int(cfg.get("atr_window", 14))
    trend_win = int(cfg.get("trend_window", 50))
    q_hi      = float(cfg.get("vol_high_pct", 0.9))
    q_lo      = float(cfg.get("vol_low_pct", 0.5))

    # 계산
    atr = _atr(df["high"], df["low"], df["close"], window=atr_win)
    slope = _trend_slope(df["close"], window=trend_win)

    # 분위 기반 변동성 구간
    thr_hi = float(atr.quantile(q_hi)) if len(atr) else 0.0
    thr_lo = float(atr.quantile(q_lo)) if len(atr) else 0.0
    last_atr = float(atr.iloc[-1]) if len(atr) else 0.0

    if last_atr >= thr_hi:
        vol_reg = 2
    elif last_atr <= thr_lo:
        vol_reg = 0
    else:
        vol_reg = 1

    # 추세 구간
    last_slope = float(slope.iloc[-1]) if len(slope) else 0.0
    if last_slope > 0:
        trend_reg = 2
    elif last_slope < 0:
        trend_reg = 0
    else:
        trend_reg = 1

    tag = int(vol_reg * 3 + trend_reg)

    out = {
        "enabled": True,
        "vol_regime": vol_reg,
        "trend_regime": trend_reg,
        "regime_tag": tag,
        "stats": {
            "atr_window": atr_win,
            "trend_window": trend_win,
            "atr_q_low": thr_lo,
            "atr_q_high": thr_hi,
            "atr_last": last_atr,
            "slope_last": last_slope,
            "rows": int(len(df))
        }
    }
    _CACHE[key] = (t, out)
    return out
