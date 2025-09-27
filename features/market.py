# features/market.py
# (2025-09-27) BTC 도미넌스/토탈·알트 인덱스, 펀딩·베이시스, 파생 파생지표(z-score) 생성
import os, json
import numpy as np
import pandas as pd

def _to_df(x):
    if x is None: return None
    if isinstance(x, pd.DataFrame): return x
    if isinstance(x, str) and os.path.exists(x):
        try: return pd.read_csv(x, encoding="utf-8-sig")
        except Exception: return pd.read_csv(x)
    return None

def _align_on_ts(df, refs: dict):
    out = {}
    for k, v in refs.items():
        d = _to_df(v)
        if d is None or "timestamp" not in d.columns: 
            out[k] = None; continue
        dd = d.copy()
        dd["timestamp"] = pd.to_datetime(dd["timestamp"], errors="coerce")
        out[k] = dd.set_index("timestamp").sort_index()
    base = df.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], errors="coerce")
    base = base.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return base, out

def _z(x, win=96):
    s = pd.Series(x, dtype=float)
    m = s.rolling(win, min_periods=max(10, win//5)).mean()
    v = s.rolling(win, min_periods=max(10, win//5)).std()
    return ((s - m) / (v.replace(0, np.nan))).fillna(0.0).values

def compute_market_features(symbol: str, df: pd.DataFrame, strategy: str,
                            loaders: dict | None = None) -> pd.DataFrame:
    """
    loaders (optional):
      - "BTCUSDT": DataFrame or CSV path (timestamp, close, volume)
      - "TOTAL"  : total mkt index (timestamp, close)
      - "ALTS"   : alt mkt index (timestamp, close)
      - "FUNDING": funding rate for symbol (timestamp, value)
      - "BASIS"  : futures basis for symbol (timestamp, value)
    """
    loaders = loaders or {}
    base, refs = _align_on_ts(df, loaders)

    feat = pd.DataFrame(index=base.index)
    # ----- price returns (own)
    close = base["close"].astype(float)
    feat["ret_1h"]  = close.pct_change().fillna(0.0)
    feat["ret_4h"]  = close.pct_change(4).fillna(0.0)
    feat["ret_24h"] = close.pct_change(24).fillna(0.0)

    # ----- BTC dominance proxy & returns
    btc = refs.get("BTCUSDT")
    if btc is not None and "close" in btc.columns:
        btc_close = btc["close"].astype(float).reindex(feat.index).ffill()
        feat["btc_ret_24h"] = btc_close.pct_change(24).fillna(0.0)
        feat["pair_ret_spread_24h"] = (feat["ret_24h"] - feat["btc_ret_24h"]).fillna(0.0)
    else:
        feat["btc_ret_24h"] = 0.0
        feat["pair_ret_spread_24h"] = 0.0

    # ----- Total / Alt index
    tot = refs.get("TOTAL"); alts = refs.get("ALTS")
    if tot is not None and "close" in tot.columns:
        t = tot["close"].astype(float).reindex(feat.index).ffill()
        feat["total_ret_24h"] = t.pct_change(24).fillna(0.0)
    else:
        feat["total_ret_24h"] = 0.0
    if alts is not None and "close" in alts.columns:
        a = alts["close"].astype(float).reindex(feat.index).ffill()
        feat["alts_ret_24h"] = a.pct_change(24).fillna(0.0)
    else:
        feat["alts_ret_24h"] = 0.0

    # dominance proxy: (ALTS / TOTAL), z-score
    try:
        dom = (a / (t.replace(0, np.nan))).clip(upper=np.inf)
        feat["alts_share_z"] = pd.Series(_z(dom.values, 96), index=feat.index)
    except Exception:
        feat["alts_share_z"] = 0.0

    # ----- Funding / Basis (이미 정렬된 시계열 가정, 없으면 0)
    for key, col in [("FUNDING", "funding_rate"), ("BASIS", "basis")]:
        d = refs.get(key)
        if d is not None:
            # 'value' 컬럼 또는 명시 컬럼 사용
            if "value" in d.columns:
                v = d["value"].astype(float)
            elif col in d.columns:
                v = d[col].astype(float)
            else:
                v = pd.Series(0.0, index=d.index)
            feat[key.lower()] = v.reindex(feat.index).fillna(method="ffill").fillna(0.0)
        else:
            feat[key.lower()] = 0.0

    # ----- Z-scored versions for stationarity
    for c in ["ret_1h","ret_4h","ret_24h","btc_ret_24h","pair_ret_spread_24h",
              "total_ret_24h","alts_ret_24h","funding","basis"]:
        try:
            feat[c+"_z"] = _z(feat[c].values, 96)
        except Exception:
            feat[c+"_z"] = 0.0

    feat = feat.reset_index().rename(columns={"index":"timestamp"})
    return feat
