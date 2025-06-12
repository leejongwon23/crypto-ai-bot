# ✅ Render 캐시 강제 무효화용 주석 — 절대 삭제하지 마

import requests
import pandas as pd
import numpy as np
import time
import pytz

BASE_URL = "https://api.bybit.com"
BTC_DOMINANCE_CACHE = {"value": 0.5, "timestamp": 0}

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "TRXUSDT",
    "LTCUSDT", "BCHUSDT", "LINKUSDT", "ATOMUSDT", "XLMUSDT",
    "ETCUSDT", "ICPUSDT", "HBARUSDT", "FILUSDT", "SANDUSDT",
    "RNDRUSDT", "INJUSDT", "NEARUSDT", "APEUSDT", "ARBUSDT",
    "AAVEUSDT", "OPUSDT", "SUIUSDT", "DYDXUSDT", "CHZUSDT",
    "LDOUSDT", "STXUSDT", "GMTUSDT", "FTMUSDT", "WLDUSDT",
    "TIAUSDT", "SEIUSDT", "ARKMUSDT", "JASMYUSDT", "AKTUSDT",
    "GMXUSDT", "SKLUSDT", "BLURUSDT", "ENSUSDT", "CFXUSDT",
    "FLOWUSDT", "ALGOUSDT", "MINAUSDT", "NEOUSDT", "MASKUSDT",
    "KAVAUSDT", "BATUSDT", "ZILUSDT", "WAVESUSDT", "OCEANUSDT",
    "1INCHUSDT", "YFIUSDT", "STGUSDT", "GALAUSDT", "IMXUSDT"
]

STRATEGY_CONFIG = {
    "단기": {"interval": "240", "limit": 600},  # 4시간봉, 최대 1000개
    "중기": {"interval": "D", "limit": 600},    # 1일봉, 최대 1000개
    "장기": {"interval": "D", "limit": 600}     # 1일봉, 최대 1000개
}

def get_btc_dominance():
    global BTC_DOMINANCE_CACHE
    now = time.time()
    if now - BTC_DOMINANCE_CACHE["timestamp"] < 1800:
        return BTC_DOMINANCE_CACHE["value"]
    try:
        url = "https://api.coinpaprika.com/v1/global"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        dom = float(data["bitcoin_dominance_percentage"]) / 100
        BTC_DOMINANCE_CACHE = {"value": round(dom, 4), "timestamp": now}
        return BTC_DOMINANCE_CACHE["value"]
    except:
        return BTC_DOMINANCE_CACHE["value"]

import numpy as np

def create_dataset(features, window=20, strategy="단기"):
    import numpy as np

    X, y = [], []
    if not features or len(features) <= window:
        print(f"[스킵] features 부족 → len={len(features)}")
        return np.array([]), np.array([])

    columns = [c for c in features[0].keys() if c != "timestamp"]
    class_ranges = [
        (-0.30, -0.15), (-0.15, -0.10), (-0.10, -0.07), (-0.07, -0.05),
        (-0.05, -0.03), (-0.03, -0.015), (-0.015, -0.01),
        ( 0.01,  0.015), ( 0.015, 0.03), ( 0.03, 0.05), ( 0.05, 0.07),
        ( 0.07, 0.10), ( 0.10, 0.15), ( 0.15, 0.20), ( 0.20, 0.30),
        ( 0.30, 0.40), ( 0.40, 0.60)
    ]
    strategy_minutes = {"단기": 240, "중기": 1440, "장기": 10080}
    lookahead_minutes = strategy_minutes.get(strategy, 1440)

    for i in range(window, len(features) - 3):
        try:
            seq = features[i - window:i]
            base = features[i]
            entry_time = base.get("timestamp")
            entry_price = float(base.get("close", 0.0))
            if not entry_time or entry_price <= 0: continue

            future = [f for f in features[i + 1:]
                      if f.get("timestamp") and (f["timestamp"] - entry_time).total_seconds() / 60 <= lookahead_minutes]
            if len(seq) != window or len(future) < 1: continue

            max_future_price = max(f.get("high", f.get("close", entry_price)) for f in future)
            gain = (max_future_price - entry_price) / (entry_price + 1e-6)
            if not np.isfinite(gain) or abs(gain) > 5: continue

            cls = next((j for j, (low, high) in enumerate(class_ranges) if low <= gain < high), -1)
            if cls == -1: continue

            sample = [[float(r.get(c, 0.0)) for c in columns] for r in seq]
            if any(len(row) != len(columns) for row in sample): continue

            X.append(sample)
            y.append(cls)

        except Exception as e:
            continue

    print(f"[확인] 학습 샘플 생성 완료 → {len(X)}개")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def get_kline_by_strategy(symbol: str, strategy: str):
    config = STRATEGY_CONFIG.get(strategy)
    if config is None:
        print(f"[오류] 전략 설정 없음: {strategy}")
        return None

    df = get_kline(symbol, interval=config["interval"], limit=config["limit"])
    
    if df is None or df.empty:
        print(f"[경고] {symbol}-{strategy}: get_kline_by_strategy() → 데이터 없음")
    else:
        print(f"[확인] {symbol}-{strategy}: 데이터 {len(df)}개 확보")

    return df

def get_kline(symbol: str, interval: str = "60", limit: int = 300) -> pd.DataFrame:
    """
    Bybit Kline 데이터를 불러오는 함수
    :param symbol: 종목명 (예: BTCUSDT)
    :param interval: 시간 간격 ("60"=1시간, "240"=4시간, "D"=1일)
    :param limit: 캔들 개수 (기본 300개)
    :return: DataFrame (timestamp, open, high, low, close, volume)
    """
    try:
        url = f"{BASE_URL}/v5/market/kline"
        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if "result" not in data or "list" not in data["result"]:
            print(f"[경고] get_kline() → 데이터 응답 구조 이상")
            return None

        raw = data["result"]["list"]
        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["datetime"] = df["timestamp"]
        return df

    except Exception as e:
        print(f"[에러] get_kline({symbol}) 실패 → {e}")
        return None


def get_realtime_prices():
    url = f"{BASE_URL}/v5/market/tickers"
    params = {"category": "linear"}
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if "result" not in data or "list" not in data["result"]:
            return {}
        tickers = data["result"]["list"]
        return {item["symbol"]: float(item["lastPrice"]) for item in tickers if item["symbol"] in SYMBOLS}
    except:
        return {}

def compute_features(symbol: str, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    df = df.copy()

    # ✅ timestamp 보장
    if "datetime" in df.columns:
        df["timestamp"] = df["datetime"]
    elif "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime("now")

    # ✅ 신뢰성 높은 기본 지표
    df['ma20'] = df['close'].rolling(window=20).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    df['rsi'] = 100 - (100 / (1 + rs))

    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow

    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    df['trend_score'] = df['close'].pct_change(periods=3)

    # ✅ Stochastic RSI
    min_rsi = df['rsi'].rolling(14).min()
    max_rsi = df['rsi'].rolling(14).max()
    df['stoch_rsi'] = (df['rsi'] - min_rsi) / (max_rsi - min_rsi + 1e-6)

    # ✅ CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    df['cci'] = cci

    # ✅ OBV
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv

    # ✅ 볼린저 밴드
    bb_ma = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bollinger'] = (df['close'] - bb_ma) / (2 * bb_std + 1e-6)

    # ✅ 전략별 특화 피처
    if strategy == "중기":
        df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_cross'] = df['ema5'] - df['ema20']

    elif strategy == "장기":
        df['volume_cumsum'] = df['volume'].cumsum()
        df['roc'] = df['close'].pct_change(periods=12)
        mf = df["close"] * df["volume"]
        pos_mf = mf.where(df["close"] > df["close"].shift(), 0)
        neg_mf = mf.where(df["close"] < df["close"].shift(), 0)
        mf_ratio = pos_mf.rolling(14).sum() / (neg_mf.rolling(14).sum() + 1e-6)
        df["mfi"] = 100 - (100 / (1 + mf_ratio))

    # ✅ 피처 선택
    base = [
        "timestamp", "close", "volume", "ma20", "rsi", "macd", "bollinger",
        "volatility", "trend_score", "stoch_rsi", "cci", "obv"
    ]
    mid_extra = ["ema_cross"]
    long_extra = ["volume_cumsum", "roc", "mfi"]

    if strategy == "중기":
        base += mid_extra
    elif strategy == "장기":
        base += long_extra

    df = df[base].dropna()
    print(f"[완료] {symbol}-{strategy}: 피처 {df.shape[0]}개 생성")
    return df
