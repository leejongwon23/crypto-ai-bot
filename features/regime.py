# features/regime.py
# (2025-09-27) 변동성·추세 레짐 지표 및 간단 국면 분류
import numpy as np
import pandas as pd

def _ema(x: pd.Series, span: int):
    return x.ewm(span=span, adjust=False).mean()

def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    변동성·추세 기반 레짐 피처 생성
    반환: timestamp + 레짐 관련 컬럼들
    """
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    close = d["close"].astype(float)

    # --- 변동성: 실현 변동성 (24h vs 96h), z-score
    ret = close.pct_change().fillna(0.0)
    vol_24 = ret.rolling(24, min_periods=8).std().fillna(method="ffill").fillna(0.0)
    vol_96 = ret.rolling(96, min_periods=24).std().fillna(method="ffill").fillna(0.0)
    vol_z  = ((vol_24 - vol_96.rolling(96, min_periods=24).mean()) /
              (vol_96.rolling(96, min_periods=24).std().replace(0, np.nan))).fillna(0.0)

    # --- 추세: EMA 크로스 + 기울기
    ema_fast = _ema(close, 20)
    ema_slow = _ema(close, 60)
    slope = pd.Series(np.gradient(close.values), index=close.index) / (close.rolling(20).mean() + 1e-12)
    trend = (ema_fast - ema_slow) / (close + 1e-12)

    # --- 국면 라벨
    bull = ((trend > 0) & (slope > 0)).astype(int)
    bear = ((trend < 0) & (slope < 0)).astype(int)
    chop = (1 - ((bull + bear).clip(upper=1))).astype(int)

    vol_hi = (vol_z > 0.5).astype(int)
    vol_lo = (vol_z < -0.5).astype(int)

    out = pd.DataFrame({
        "vol_24": vol_24,
        "vol_96": vol_96,
        "vol_z": vol_z,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "trend": trend,
        "slope": slope,
        "reg_bull": bull,
        "reg_bear": bear,
        "reg_chop": chop,
        "reg_vol_hi": vol_hi,
        "reg_vol_lo": vol_lo,
    }, index=close.index).fillna(0.0)

    return out.reset_index().rename(columns={"index":"timestamp"})
