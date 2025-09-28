# features/market.py
# (2025-09-27) 시장 컨텍스트: BTC/총시총/알트 인덱스, 펀딩/베이시스, Z-score 파생
# utils._merge_asof_all 이 기대하는 시그니처: get_market_context_df(ts, strategy, symbol=None)
import os
import pandas as pd
import numpy as np

# ----- 경로 기본값(있으면 사용, 없으면 0으로 대체) -----
CTX_DIR = "/persistent/context"
BTC_PATH   = os.path.join(CTX_DIR, "btc.csv")      # columns: timestamp, close, volume(optional)
TOTAL_PATH = os.path.join(CTX_DIR, "total.csv")    # columns: timestamp, close
ALTS_PATH  = os.path.join(CTX_DIR, "alts.csv")     # columns: timestamp, close
FUND_DIR   = os.path.join(CTX_DIR, "funding")      # per-symbol csv: columns: timestamp, value or funding_rate
BASIS_DIR  = os.path.join(CTX_DIR, "basis")        # per-symbol csv: columns: timestamp, value or basis

def _read_csv(path: str) -> pd.DataFrame | None:
    try:
        if not os.path.exists(path):
            return None
        # UTF-8-SIG 우선, 실패 시 기본 인코딩
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            return None
        df = df.copy()
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = ts.dt.tz_convert("Asia/Seoul")
        # 표준 컬럼 이름 정리
        if "close" not in df.columns and "value" in df.columns:
            df = df.rename(columns={"value": "close"})
        return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    except Exception:
        return None

def _load_ref_series():
    btc   = _read_csv(BTC_PATH)
    total = _read_csv(TOTAL_PATH)
    alts  = _read_csv(ALTS_PATH)
    return btc, total, alts

def _load_symbol_series(symbol: str | None):
    def _first_existing(paths):
        for p in paths:
            if os.path.exists(p):
                df = _read_csv(p)
                if df is not None:
                    return df
        return None
    symbol = (symbol or "").upper()
    funding = _first_existing([
        os.path.join(FUND_DIR, f"{symbol}.csv"),
        os.path.join(FUND_DIR, f"{symbol.lower()}.csv"),
    ])
    basis = _first_existing([
        os.path.join(BASIS_DIR, f"{symbol}.csv"),
        os.path.join(BASIS_DIR, f"{symbol.lower()}.csv"),
    ])
    return funding, basis

def _median_step(ts: pd.Series) -> pd.Timedelta:
    """시계열 간격(중앙값) 추정 -> 24h 변화 계산에 사용"""
    t = pd.to_datetime(ts, errors="coerce")
    d = t.sort_values().diff().dropna()
    if d.empty:
        return pd.Timedelta(hours=1)
    # 비정상 큰 간격 제외(상위 5% 절단)
    q95 = d.quantile(0.95)
    d = d[d <= q95]
    return (d.median() if not d.empty else pd.Timedelta(hours=1))

def _zscore(arr: pd.Series | np.ndarray, win: int = 96) -> np.ndarray:
    s = pd.Series(arr, dtype=float)
    m = s.rolling(win, min_periods=max(10, win // 5)).mean()
    v = s.rolling(win, min_periods=max(10, win // 5)).std()
    z = (s - m) / v.replace(0, np.nan)
    return z.fillna(0.0).to_numpy(dtype=float)

def _pct_change_by_hours(series: pd.Series, ts_index: pd.DatetimeIndex, hours: int) -> pd.Series:
    """
    샘플 간 간격을 추정해 'hours'에 해당하는 step으로 pct_change 계산
    """
    if series.empty:
        return pd.Series(0.0, index=ts_index)
    step = _median_step(ts_index)
    # 최소 1 step 보장
    periods = int(max(1, round(pd.Timedelta(hours=hours) / step)))
    return series.pct_change(periods).reindex(ts_index).fillna(0.0)

def _reindex_ffill(df: pd.DataFrame | None, ts_index: pd.DatetimeIndex, value_col: str = "close") -> pd.Series:
    if df is None or value_col not in df.columns:
        return pd.Series(0.0, index=ts_index, dtype=float)
    s = pd.to_numeric(df[value_col], errors="coerce").astype(float)
    s.index = pd.to_datetime(df["timestamp"], errors="coerce")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(ts_index).ffill().fillna(0.0)

def get_market_context_df(ts: pd.Series, strategy: str, symbol: str | None = None) -> pd.DataFrame:
    """
    utils._merge_asof_all 에 의해 base와 asof-merge 됩니다.
    반환: DataFrame[timestamp, btc_ret_24h, total_ret_24h, alts_ret_24h,
                    alts_share_z, funding, basis,
                    btc_ret_24h_z, total_ret_24h_z, alts_ret_24h_z, funding_z, basis_z]
    데이터가 없으면 0으로 채워진 동일 길이 프레임을 반환.
    """
    # 1) 타임스탬프 표준화
    ts = pd.to_datetime(ts, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")
    ts = ts.dropna()
    if ts.empty:
        return pd.DataFrame(columns=["timestamp"])

    idx = pd.DatetimeIndex(ts)

    # 2) 외부 참조 로딩 (있으면 사용)
    btc, total, alts = _load_ref_series()
    funding_df, basis_df = _load_symbol_series(symbol)

    # 3) 시계열 재색인 및 변화율 계산
    btc_close   = _reindex_ffill(btc, idx, "close")
    total_close = _reindex_ffill(total, idx, "close")
    alts_close  = _reindex_ffill(alts, idx, "close")

    btc_ret_24h   = _pct_change_by_hours(btc_close,   idx, 24)
    total_ret_24h = _pct_change_by_hours(total_close, idx, 24)
    alts_ret_24h  = _pct_change_by_hours(alts_close,  idx, 24)

    # 알트 점유율(대용): ALTS / TOTAL 의 z-score
    with np.errstate(divide="ignore", invalid="ignore"):
        share = (alts_close / total_close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    alts_share_z = pd.Series(_zscore(share, 96), index=idx)

    # 펀딩/베이시스 (심볼별, 없으면 0)
    # funding: 'value' 또는 'funding_rate' 컬럼 허용
    def _pick_value(df: pd.DataFrame | None, pref1: str, pref2: str) -> pd.Series:
        if df is None:
            return pd.Series(0.0, index=idx, dtype=float)
        col = pref1 if pref1 in df.columns else (pref2 if pref2 in df.columns else None)
        if col is None:
            return pd.Series(0.0, index=idx, dtype=float)
        return _reindex_ffill(df.rename(columns={col: "val"}), idx, "val")

    funding = _pick_value(funding_df, "value", "funding_rate")
    basis   = _pick_value(basis_df,   "value", "basis")

    # 4) Z-score 파생
    btc_ret_24h_z   = pd.Series(_zscore(btc_ret_24h,   96), index=idx)
    total_ret_24h_z = pd.Series(_zscore(total_ret_24h, 96), index=idx)
    alts_ret_24h_z  = pd.Series(_zscore(alts_ret_24h,  96), index=idx)
    funding_z       = pd.Series(_zscore(funding,       96), index=idx)
    basis_z         = pd.Series(_zscore(basis,         96), index=idx)

    # 5) 결과 프레임
    out = pd.DataFrame({
        "timestamp": idx,
        "btc_ret_24h":   btc_ret_24h.values,
        "total_ret_24h": total_ret_24h.values,
        "alts_ret_24h":  alts_ret_24h.values,
        "alts_share_z":  alts_share_z.values,
        "funding":       funding.values,
        "basis":         basis.values,
        "btc_ret_24h_z":   btc_ret_24h_z.values,
        "total_ret_24h_z": total_ret_24h_z.values,
        "alts_ret_24h_z":  alts_ret_24h_z.values,
        "funding_z":       funding_z.values,
        "basis_z":         basis_z.values,
    })
    return out
