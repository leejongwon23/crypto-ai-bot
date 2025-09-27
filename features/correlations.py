# features/correlations.py
# (2025-09-27) 코인-코인 롤링 상관/β, EWMA-β, 간단 RLS(칼만풍) β 추정
import numpy as np
import pandas as pd

def _align(a: pd.Series, b: pd.Series):
    s = pd.concat([a, b], axis=1).dropna()
    return s.iloc[:,0], s.iloc[:,1]

def rolling_corr_beta(asset_close: pd.Series,
                      bench_close: pd.Series,
                      win: int = 96) -> pd.DataFrame:
    a, b = _align(asset_close.pct_change().fillna(0.0),
                  bench_close.pct_change().fillna(0.0))
    cov = a.rolling(win, min_periods=max(10, win//5)).cov(b)
    var = b.rolling(win, min_periods=max(10, win//5)).var()
    beta = (cov / var.replace(0, np.nan)).fillna(0.0)
    corr = a.rolling(win, min_periods=max(10, win//5)).corr(b).fillna(0.0)
    out = pd.DataFrame({"beta_roll": beta, "corr_roll": corr})
    return out.reindex(asset_close.index).fillna(method="ffill").fillna(0.0)

def ewma_beta(asset_close: pd.Series,
              bench_close: pd.Series,
              span: int = 96) -> pd.Series:
    r_a = asset_close.pct_change().fillna(0.0)
    r_b = bench_close.pct_change().fillna(0.0)
    cov = (r_a * r_b).ewm(span=span, adjust=False).mean()
    var = (r_b ** 2).ewm(span=span, adjust=False).mean()
    beta = (cov / var.replace(0, np.nan)).fillna(0.0)
    return beta.reindex(asset_close.index).fillna(method="ffill").fillna(0.0)

def rls_beta(asset_close: pd.Series,
             bench_close: pd.Series,
             lam: float = 0.99, delta: float = 1000.0) -> pd.Series:
    """
    간단 RLS(Recursive Least Squares) — 1차원 베타 추정 (칼만풍)
    """
    r_a = asset_close.pct_change().fillna(0.0)
    r_b = bench_close.pct_change().fillna(0.0)
    idx = r_a.index
    theta = 0.0
    P = delta
    betas = []
    for t in idx:
        x = float(r_b.loc[t]); y = float(r_a.loc[t])
        # 예측 오차
        e = y - theta * x
        # 칼만 이득
        K = (P * x) / (lam + x * P * x)
        # 갱신
        theta = theta + K * e
        P = (P - K * x * P) / lam
        betas.append(theta)
    s = pd.Series(betas, index=idx, dtype=float)
    return s.reindex(asset_close.index).fillna(method="ffill").fillna(0.0)

def build_correlation_block(asset_df: pd.DataFrame,
                            bench_df: pd.DataFrame,
                            win: int = 96) -> pd.DataFrame:
    a = asset_df.copy(); b = bench_df.copy()
    a["timestamp"] = pd.to_datetime(a["timestamp"], errors="coerce")
    b["timestamp"] = pd.to_datetime(b["timestamp"], errors="coerce")
    a = a.set_index("timestamp").sort_index()
    b = b.set_index("timestamp").sort_index()
    rc = rolling_corr_beta(a["close"].astype(float), b["close"].astype(float), win=win)
    rc["beta_ewma"] = ewma_beta(a["close"].astype(float), b["close"].astype(float), span=win)
    rc["beta_rls"]  = rls_beta(a["close"].astype(float), b["close"].astype(float))
    return rc.reset_index().rename(columns={"index":"timestamp"})
