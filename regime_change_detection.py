# regime_change_detection.py

import numpy as np
import pandas as pd

def detect_volatility_change(df, window=20, threshold=2.0):
    """
    df: dataframe with 'close' column
    window: rolling window size
    threshold: std change multiple to trigger detection
    """
    df = df.copy()
    df['volatility'] = df['close'].pct_change().rolling(window).std()

    recent_vol = df['volatility'].iloc[-1]
    mean_vol = df['volatility'].mean()
    std_vol = df['volatility'].std()

    if recent_vol > mean_vol + threshold * std_vol:
        print(f"[Regime Detection] 변동성 급등 감지 → recent: {recent_vol:.4f}, mean: {mean_vol:.4f}")
        return True

    return False

def detect_mean_shift(df, window=20, threshold=2.0):
    """
    df: dataframe with 'close' column
    Detects if recent mean deviates from global mean by threshold*std
    """
    df = df.copy()
    df['ma'] = df['close'].rolling(window).mean()

    recent_ma = df['ma'].iloc[-1]
    global_mean = df['close'].mean()
    global_std = df['close'].std()

    if abs(recent_ma - global_mean) > threshold * global_std:
        print(f"[Regime Detection] 평균 급변 감지 → recent_ma: {recent_ma:.4f}, global_mean: {global_mean:.4f}")
        return True

    return False

def detect_regime_change(df):
    """
    Main detection function combining volatility and mean shift
    """
    vol_change = detect_volatility_change(df)
    mean_shift = detect_mean_shift(df)

    if vol_change or mean_shift:
        print("[Regime Detection] 시장 변화 감지됨 → 모델 재학습 권장")
        return True

    print("[Regime Detection] 시장 안정 → 모델 유지")
    return False
