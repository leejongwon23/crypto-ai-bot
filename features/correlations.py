# features/correlations.py
# (2025-09-27) 코인-벤치( BTC/ETH ) 롤링 상관 & β (roll/EWMA/RLS) 블록 생성
import os
import numpy as np
import pandas as pd

# --------- 내부 기본 파라미터 ---------
_DEF_WIN = int(os.getenv("CORR_WIN", "96"))   # 캔들 기준 윈도우(기본 96개)
_MINP    = lambda w: max(10, w // 5)

# --------- 저수준 계산 유틸 ---------
def _align(a: pd.Series, b: pd.Series):
    s = pd.concat([a, b], axis=1).dropna()
    return s.iloc[:, 0], s.iloc[:, 1]

def _rolling_corr_beta(asset_close: pd.Series, bench_close: pd.Series, win: int) -> pd.DataFrame:
    a, b = _align(asset_close.pct_change().fillna(0.0), bench_close.pct_change().fillna(0.0))
    cov  = a.rolling(win, min_periods=_MINP(win)).cov(b)
    var  = b.rolling(win, min_periods=_MINP(win)).var()
    beta = (cov / var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = a.rolling(win, min_periods=_MINP(win)).corr(b).fillna(0.0)
    return pd.DataFrame({"beta_roll": beta, "corr_roll": corr})

def _ewma_beta(asset_close: pd.Series, bench_close: pd.Series, span: int) -> pd.Series:
    r_a = asset_close.pct_change().fillna(0.0)
    r_b = bench_close.pct_change().fillna(0.0)
    cov = (r_a * r_b).ewm(span=span, adjust=False).mean()
    var = (r_b ** 2).ewm(span=span, adjust=False).mean()
    beta = (cov / var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return beta

def _rls_beta(asset_close: pd.Series, bench_close: pd.Series, lam: float = 0.99, delta: float = 1000.0) -> pd.Series:
    """1차원 RLS(칼만풍) 베타 추정"""
    r_a = asset_close.pct_change().fillna(0.0)
    r_b = bench_close.pct_change().fillna(0.0)
    theta, P = 0.0, float(delta)
    out = []
    for x, y in zip(r_b.values.astype(float), r_a.values.astype(float)):
        e = y - theta * x
        K = (P * x) / (lam + x * P * x)
        theta = theta + K * e
        P = (P - K * x * P) / lam
        out.append(theta)
    s = pd.Series(out, index=r_a.index, dtype=float)
    return s

# --------- 메인: utils가 호출하는 컨텍스트 생성기 ---------
def get_rolling_corr_df(symbol: str, ts: pd.Series, strategy: str) -> pd.DataFrame:
    """
    반환: DataFrame[timestamp,
                    corr_btc, beta_btc_roll, beta_btc_ewma, beta_btc_rls,
                    corr_eth, beta_eth_roll, beta_eth_ewma, beta_eth_rls]
    데이터가 부족하면 0으로 채움.
    """
    # ts 표준화
    ts = pd.to_datetime(ts, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")
    ts = ts.dropna()
    if ts.empty:
        return pd.DataFrame(columns=["timestamp"])

    idx = pd.DatetimeIndex(ts)

    # 가격 불러오기 (우리 프로젝트 유틸)
    try:
        from data.utils import get_kline_by_strategy as _get_k
        df_a   = _get_kline_by_strategy(symbol, strategy)
        df_btc = _get_kline_by_strategy("BTCUSDT", strategy)
        df_eth = _get_kline_by_strategy("ETHUSDT", strategy)
    except Exception:
        # 없어도 안전 반환
        return pd.DataFrame({"timestamp": idx}).assign(
            corr_btc=0.0, beta_btc_roll=0.0, beta_btc_ewma=0.0, beta_btc_rls=0.0,
            corr_eth=0.0, beta_eth_roll=0.0, beta_eth_ewma=0.0, beta_eth_rls=0.0
        )

    def _prep(df):
        if df is None or df.empty or "close" not in df.columns or "timestamp" not in df.columns:
            return pd.Series(0.0, index=idx)
        s = pd.to_numeric(df["close"], errors="coerce").astype(float)
        s.index = pd.to_datetime(df["timestamp"], errors="coerce")
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s.reindex(idx).ffill().fillna(method="bfill")

    a   = _prep(df_a)
    btc = _prep(df_btc)
    eth = _prep(df_eth)

    win = int(os.getenv("CORR_WIN", str(_DEF_WIN)))

    # BTC 기준 블록
    blk_btc = _rolling_corr_beta(a, btc, win=win)
    blk_btc["beta_ewma"] = _ewma_beta(a, btc, span=win).reindex(idx).fillna(method="ffill").fillna(0.0)
    blk_btc["beta_rls"]  = _rls_beta(a, btc).reindex(idx).fillna(method="ffill").fillna(0.0)
    blk_btc = blk_btc.reindex(idx).fillna(method="ffill").fillna(0.0)

    # ETH 기준 블록
    blk_eth = _rolling_corr_beta(a, eth, win=win)
    blk_eth["beta_ewma"] = _ewma_beta(a, eth, span=win).reindex(idx).fillna(method="ffill").fillna(0.0)
    blk_eth["beta_rls"]  = _rls_beta(a, eth).reindex(idx).fillna(method="ffill").fillna(0.0)
    blk_eth = blk_eth.reindex(idx).fillna(method="ffill").fillna(0.0)

    out = pd.DataFrame({
        "timestamp": idx,
        "corr_btc":      blk_btc["corr_roll"].values,
        "beta_btc_roll": blk_btc["beta_roll"].values,
        "beta_btc_ewma": blk_btc["beta_ewma"].values,
        "beta_btc_rls":  blk_btc["beta_rls"].values,
        "corr_eth":      blk_eth["corr_roll"].values,
        "beta_eth_roll": blk_eth["beta_roll"].values,
        "beta_eth_ewma": blk_eth["beta_ewma"].values,
        "beta_eth_rls":  blk_eth["beta_rls"].values,
    })
    return out
