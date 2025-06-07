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
    "단기": {"interval": "60", "limit": 300},    # 1시간봉
    "중기": {"interval": "240", "limit": 300},  # 4시간봉
    "장기": {"interval": "D", "limit": 300}     # 1일봉
}

DEFAULT_MIN_GAIN = {
    "단기": 0.01,
    "중기": 0.03,
    "장기": 0.05
}

def get_min_gain(symbol: str, strategy: str):
    try:
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < 10:
            return DEFAULT_MIN_GAIN[strategy]
        df = df.tail(7)
        pct_changes = df["close"].pct_change().abs()
        avg_volatility = pct_changes.mean()
        return max(round(avg_volatility, 4), DEFAULT_MIN_GAIN[strategy])
    except:
        return DEFAULT_MIN_GAIN[strategy]

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
    import pandas as pd
    from collections import Counter
    import random

    X, y = []

    if not features or len(features) <= window:
        print(f"[스킵] features 부족 → len={len(features)}")
        return np.array([]), np.array([])

    try:
        columns = [c for c in features[0].keys() if c != "timestamp"]
    except Exception as e:
        print(f"[오류] features[0] 키 확인 실패 → {e}")
        return np.array([]), np.array([])

    bins = [-0.15, -0.10, -0.07, -0.05, -0.03, -0.015, -0.005,
             0.005, 0.015, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]
    
    strategy_minutes = {"단기": 240, "중기": 1440, "장기": 10080}
    lookahead_minutes = strategy_minutes.get(strategy, 1440)

    for i in range(window, len(features)):
        try:
            seq = features[i - window:i]
            base = features[i]
            entry_time = base.get("timestamp")
            entry_price = float(base.get("close", 0.0))

            if not entry_time or entry_price <= 0:
                continue

            future = [f for f in features[i + 1:] if f.get("timestamp") and (f["timestamp"] - entry_time).total_seconds() / 60 <= lookahead_minutes]
            if len(seq) != window or len(future) < 2:
                continue

            highs = [float(f.get("high", entry_price)) for f in future]
            lows = [float(f.get("low", entry_price)) for f in future]
            if not highs or not lows or any(pd.isna(highs)) or any(pd.isna(lows)):
                continue

            max_gain = max((h - entry_price) / (entry_price + 1e-6) for h in highs)
            max_loss = max((entry_price - l) / (entry_price + 1e-6) for l in lows)
            if not np.isfinite(max_gain) or not np.isfinite(max_loss):
                continue

            if abs(max_gain) > 0.6 or abs(max_loss) > 0.6:
                continue

            direction = "롱" if max_gain >= max_loss else "숏"
            gain = max_gain if direction == "롱" else -max_loss

            base_cls = next((i for i, b in enumerate(bins) if gain < b), len(bins) - 1)
            cls = max(0, 7 - base_cls) if direction == "숏" else min(15, 8 + base_cls)

            if not (0 <= cls < 16):
                continue

            sample = [[float(r.get(c, 0.0)) for c in columns] for r in seq]
            if any(len(row) != len(columns) for row in sample):
                continue

            X.append(sample)
            y.append(cls)

        except Exception as e:
            print(f"[예외] 샘플 생성 실패 (i={i}) → {type(e).__name__}: {e}")
            continue

    if not X or not y:
        print(f"[결과 없음] 샘플 부족 → X={len(X)}, y={len(y)}")
        return np.array([]), np.array([])

    dist = Counter(y)
    total = len(y)
    print(f"[분포] 클래스 수: {len(dist)} / 총 샘플: {total}")
    for k in sorted(dist):
        print(f" · 클래스 {k:2d}: {dist[k]}개 ({dist[k]/total:.2%})")

    dominant_class, dominant_count = dist.most_common(1)[0]
    if len(dist) <= 3 or dominant_count > total * 0.85:
        print(f"⚠️ 클래스 심각 편향 감지 → 보정 시작")
        min_count = max(10, int(total * 0.04))
        for cls in range(16):
            count = dist.get(cls, 0)
            if count == 0:
                continue
            samples = [(x, y_val) for x, y_val in zip(X, y) if y_val == cls]
            while count < min_count:
                x_dup, y_dup = random.choice(samples)
                X.append(x_dup)
                y.append(y_dup)
                count += 1
        print(f"  └ 보정 완료: 클래스 다양성 확보")

    return np.array(X), np.array(y)


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

    # ✅ timestamp 복원 보장
    if "datetime" in df.columns:
        df["timestamp"] = df["datetime"]
    elif "timestamp" not in df.columns:
        print(f"[경고] {symbol}-{strategy}: timestamp 복원 불가 — datetime 없음")
        df["timestamp"] = pd.to_datetime("now")

    # ✅ 공통 기본 feature
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    df['rsi'] = 100 - (100 / (1 + rs))
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    bb_ma = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bollinger'] = (df['close'] - bb_ma) / (2 * bb_std)
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['trend_score'] = df['ema10'].pct_change()
    df['current_vs_ma20'] = (df['close'] / (df['ma20'] + 1e-6)) - 1
    df['volume_delta'] = df['volume'].diff()

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

    # ✅ CCI, Stoch RSI
    tp = (df['high'] + df['low'] + df['close']) / 3
    cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    df['cci'] = cci
    min_rsi = df['rsi'].rolling(14).min()
    max_rsi = df['rsi'].rolling(14).max()
    df['stoch_rsi'] = (df['rsi'] - min_rsi) / (max_rsi - min_rsi + 1e-6)

    # ✅ BTC 도미넌스
    btc_dom = get_btc_dominance()
    df['btc_dominance'] = btc_dom
    df['btc_dominance_diff'] = btc_dom - df['btc_dominance'].rolling(3).mean()

    # ✅ 캔들 및 거래량 특성
    df['candle_range'] = df['high'] - df['low']
    df['candle_body'] = (df['close'] - df['open']).abs()
    df['candle_body_ratio'] = df['candle_body'] / (df['candle_range'] + 1e-6)
    df['candle_upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['candle_range'] + 1e-6)
    df['candle_lower_shadow_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['candle_range'] + 1e-6)
    df['range_ratio'] = df['candle_range'] / (df['close'] + 1e-6)
    df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-6)

    # ✅ 중기: 추세 지표 (EMA cross)
    if strategy == "중기":
        df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_cross'] = df['ema5'] - df['ema20']

        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        tr = (df['high'] - df['low']).rolling(14).mean()
        dx = (abs(plus_dm - minus_dm) / (tr + 1e-6)) * 100
        df['adx'] = dx.rolling(14).mean()
        highest = df['high'].rolling(14).max()
        lowest = df['low'].rolling(14).min()
        df['willr'] = (highest - df['close']) / (highest - lowest + 1e-6) * -100

    # ✅ 장기: 거래량 기반 특성
    if strategy == "장기":
        df['volume_cumsum'] = df['volume'].cumsum()
        df['volume_cumsum_delta'] = df['volume_cumsum'].diff()
        df['volume_increase_ratio'] = df['volume'] / (df['volume'].shift(1) + 1e-6)

        mf = df["close"] * df["volume"]
        pos_mf = mf.where(df["close"] > df["close"].shift(), 0)
        neg_mf = mf.where(df["close"] < df["close"].shift(), 0)
        mf_ratio = pos_mf.rolling(14).sum() / (neg_mf.rolling(14).sum() + 1e-6)
        df["mfi"] = 100 - (100 / (1 + mf_ratio))
        df["roc"] = df["close"].pct_change(periods=12)

    # ✅ 구성
    base = [
        "timestamp", "close", "volume", "ma5", "ma20", "rsi", "macd", "bollinger", "volatility",
        "trend_score", "current_vs_ma20", "volume_delta", "obv", "cci", "stoch_rsi",
        "btc_dominance", "btc_dominance_diff",
        "candle_body_ratio", "candle_upper_shadow_ratio", "candle_lower_shadow_ratio",
        "range_ratio", "volume_ratio"
    ]
    mid_extra = ["ema_cross", "adx", "willr"]
    long_extra = ["volume_cumsum", "volume_cumsum_delta", "volume_increase_ratio", "mfi", "roc"]

    extra = []
    if strategy == "중기":
        extra = mid_extra
    elif strategy == "장기":
        extra = long_extra

    df = df[base + extra]
    df = df.dropna()

    print(f"[완료] {symbol}-{strategy}: 피처 {df.shape[0]}개 생성")
    return df
