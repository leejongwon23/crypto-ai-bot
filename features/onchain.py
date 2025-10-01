# features/onchain.py
# (2025-10-01) 온체인 컨텍스트: 활성주소/거래소유입/채굴자잔고/고래TX 등 + 24h 변동률 & Z-score
import os
import pandas as pd
import numpy as np

ONCHAIN_DIR = "/persistent/onchain"
FILES = {
    "active_addr":   ["active_addresses.csv", "active_addr.csv"],
    "exchange_in":   ["exchange_inflow.csv", "exch_inflow.csv"],
    "miner_reserve": ["miner_reserve.csv", "miners_reserve.csv"],
    "whale_tx":      ["whale_tx_count.csv", "whale_transactions.csv"],
    # 필요시 더 추가 가능: "realized_cap": ["realized_cap.csv"]
}

def _read_csv_any(path_candidates):
    for name in path_candidates:
        p = os.path.join(ONCHAIN_DIR, name)
        if os.path.exists(p):
            try:
                try:
                    df = pd.read_csv(p, encoding="utf-8-sig")
                except Exception:
                    df = pd.read_csv(p)
                if "timestamp" not in df.columns:
                    continue
                ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
                df = df.copy()
                df["timestamp"] = ts.dt.tz_convert("Asia/Seoul")
                # 값 컬럼 추론: value / close / count / amount 중 우선 존재하는 것
                for c in ["value","close","count","amount"]:
                    if c in df.columns:
                        df = df.rename(columns={c: "val"})
                        break
                if "val" not in df.columns:
                    # 숫자 단일열 추출
                    num_cols = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
                    if num_cols:
                        df = df.rename(columns={num_cols[0]: "val"})
                    else:
                        continue
                df = df[["timestamp","val"]].dropna().sort_values("timestamp").reset_index(drop=True)
                return df
            except Exception:
                continue
    return None

def _reindex_ffill(df, idx):
    if df is None:
        return pd.Series(0.0, index=idx, dtype=float)
    s = pd.to_numeric(df["val"], errors="coerce").astype(float)
    s.index = pd.to_datetime(df["timestamp"], errors="coerce")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(idx).ffill().fillna(0.0)

def _median_step(idx: pd.DatetimeIndex) -> pd.Timedelta:
    d = pd.Series(idx).sort_values().diff().dropna()
    if d.empty: return pd.Timedelta(hours=1)
    q95 = d.quantile(0.95)
    d = d[d <= q95]
    return d.median() if not d.empty else pd.Timedelta(hours=1)

def _pct_change_by_hours(series: pd.Series, idx: pd.DatetimeIndex, hours: int) -> pd.Series:
    if series.empty: return pd.Series(0.0, index=idx)
    step = _median_step(idx)
    periods = int(max(1, round(pd.Timedelta(hours=hours) / step)))
    return series.pct_change(periods).reindex(idx).fillna(0.0)

def _zscore(arr: pd.Series | np.ndarray, win: int = 96) -> np.ndarray:
    s = pd.Series(arr, dtype=float)
    m = s.rolling(win, min_periods=max(10, win//5)).mean()
    v = s.rolling(win, min_periods=max(10, win//5)).std()
    z = (s - m) / v.replace(0, np.nan)
    return z.fillna(0.0).to_numpy(dtype=float)

def get_onchain_context_df(ts: pd.Series, strategy: str, symbol: str | None = None) -> pd.DataFrame:
    # 1) ts 정규화
    ts = pd.to_datetime(ts, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
    else:
        ts = ts.dt.tz_convert("Asia/Seoul")
    ts = ts.dropna()
    if ts.empty:
        return pd.DataFrame(columns=["timestamp"])
    idx = pd.DatetimeIndex(ts)

    # 2) 온체인 개별 시리즈 로드
    aa   = _reindex_ffill(_read_csv_any(FILES["active_addr"]), idx)
    exin = _reindex_ffill(_read_csv_any(FILES["exchange_in"]), idx)
    mres = _reindex_ffill(_read_csv_any(FILES["miner_reserve"]), idx)
    wtx  = _reindex_ffill(_read_csv_any(FILES["whale_tx"]), idx)

    # 3) 파생: 24h 변동률 & z-score
    aa_ch   = _pct_change_by_hours(aa, idx, 24)
    exin_ch = _pct_change_by_hours(exin, idx, 24)
    mres_ch = _pct_change_by_hours(mres, idx, 24)
    wtx_ch  = _pct_change_by_hours(wtx, idx, 24)

    out = pd.DataFrame({
        "timestamp": idx,
        "on_active_addr": aa.values,
        "on_exchange_in": exin.values,
        "on_miner_reserve": mres.values,
        "on_whale_tx": wtx.values,
        "on_active_addr_ch_24h": aa_ch.values,
        "on_exchange_in_ch_24h": exin_ch.values,
        "on_miner_reserve_ch_24h": mres_ch.values,
        "on_whale_tx_ch_24h": wtx_ch.values,
        "on_active_addr_z": _zscore(aa, 96),
        "on_exchange_in_z": _zscore(exin, 96),
        "on_miner_reserve_z": _zscore(mres, 96),
        "on_whale_tx_z": _zscore(wtx, 96),
    })
    return out
