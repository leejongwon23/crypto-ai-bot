# features/regime.py
# (2025-09-27) 시장 레짐: 변동성(저/중/고) × 추세(하락/중립/상승) = 3x3 태그
import numpy as np
import pandas as pd
import os

# 프로젝트 설정(백분위/윈도우) 사용
try:
    from config import get_REGIME
except Exception:
    def get_REGIME():
        return {
            "enabled": True,
            "atr_window": 14,
            "trend_window": 50,
            "vol_high_pct": 0.90,
            "vol_low_pct": 0.50,
        }

def _parse_ts(ts):
    ts = pd.to_datetime(ts, errors="coerce")
    try:
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
        else:
            ts = ts.dt.tz_convert("Asia/Seoul")
    except Exception:
        pass
    return ts

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=max(3, window//3)).mean()

def _ema(x: pd.Series, span: int):
    return x.ewm(span=span, adjust=False).mean()

def _safe_close(df):
    if df is None or df.empty or "close" not in df or "timestamp" not in df:
        return pd.Series([], dtype=float)
    s = pd.to_numeric(df["close"], errors="coerce").astype(float)
    s.index = _parse_ts(df["timestamp"])
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s

def get_regime_tags_df(ts: pd.Series, strategy: str) -> pd.DataFrame:
    """
    utils._merge_asof_all 에서 호출됨.
    반환 컬럼:
      - timestamp
      - vol_regime   : 0(저) / 1(중) / 2(고)
      - trend_regime : 0(하락) / 1(중립) / 2(상승)
      - regime_tag   : vol_regime*3 + trend_regime  (0~8)
    """
    # 표준화된 타임스탬프 인덱스
    idx = pd.DatetimeIndex(_parse_ts(ts).dropna())
    if idx.size == 0:
        return pd.DataFrame(columns=["timestamp","vol_regime","trend_regime","regime_tag"])

    # 대표시장: BTCUSDT 기준으로 레짐 산출
    try:
        from data.utils import get_kline_by_strategy as _k
        df_btc = _k("BTCUSDT", strategy)
    except Exception:
        df_btc = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    close = _safe_close(df_btc).reindex(idx).ffill().bfill()
    if close.empty:
        return pd.DataFrame({"timestamp": idx, "vol_regime": 1, "trend_regime": 1, "regime_tag": 4})

    # 보조 시계열
    # high/low 가 없다면 close 로 대체
    try:
        high = pd.to_numeric(df_btc["high"], errors="coerce"); high.index = _parse_ts(df_btc["timestamp"])
        low  = pd.to_numeric(df_btc["low"],  errors="coerce"); low.index  = _parse_ts(df_btc["timestamp"])
        high = high.reindex(idx).ffill().bfill()
        low  = low.reindex(idx).ffill().bfill()
    except Exception:
        high = close.copy(); low = close.copy()

    cfg = get_REGIME()
    atr_win    = int(cfg.get("atr_window", 14))
    trend_win  = int(cfg.get("trend_window", 50))
    p_hi       = float(cfg.get("vol_high_pct", 0.90))
    p_lo       = float(cfg.get("vol_low_pct", 0.50))

    # 변동성: ATR을 백분위로 3단계
    atr = _atr(high, low, close, window=atr_win).fillna(method="ffill").fillna(0.0)
    thr_hi = float(atr.quantile(p_hi)) if np.isfinite(atr.quantile(p_hi)) else float(atr.median())
    thr_lo = float(atr.quantile(p_lo)) if np.isfinite(atr.quantile(p_lo)) else float(atr.median())
    vol_regime = np.where(atr >= thr_hi, 2, np.where(atr <= thr_lo, 0, 1)).astype(int)

    # 추세: EMA(trend_win) 기울기 기반 3단계
    ema = _ema(close, trend_win)
    # 정규화된 기울기(가격 대비 변화율)
    slope = pd.Series(np.gradient(ema.values), index=ema.index) / (close.rolling(20, min_periods=5).mean() + 1e-12)
    s_hi = float(np.nanpercentile(slope, 66))
    s_lo = float(np.nanpercentile(slope, 33))
    trend_regime = np.where(slope > s_hi, 2, np.where(slope < s_lo, 0, 1)).astype(int)

    out = pd.DataFrame({
        "timestamp": idx,
        "vol_regime": vol_regime,
        "trend_regime": trend_regime,
    })
    out["regime_tag"] = (out["vol_regime"] * 3 + out["trend_regime"]).astype(int)
    return out
