# ✅ Render 캐시 강제 무효화용 주석 — 절대 삭제하지 마
_kline_cache = {}

import requests
import pandas as pd
import numpy as np
import time
import pytz
from sklearn.preprocessing import MinMaxScaler


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

def create_dataset(features, window=20, strategy="단기", input_size=None):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from config import NUM_CLASSES

    X, y = [], []

    if not features or len(features) <= window:
        msg = f"[❌ 스킵] features 부족 → len={len(features) if features else 0}"
        print(msg)
        raise Exception(msg)

    try:
        columns = [c for c in features[0].keys() if c not in ["timestamp", "strategy"]]
    except Exception as e:
        msg = f"[오류] features[0] 키 확인 실패 → {e}"
        print(msg)
        raise Exception(msg)

    required_keys = {"timestamp", "close", "high"}
    if not all(all(k in f for k in required_keys) for f in features):
        msg = "[❌ 스킵] 필수 키 누락된 feature 존재"
        print(msg)
        raise Exception(msg)

    df = pd.DataFrame(features)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close", "high"]).sort_values("timestamp").reset_index(drop=True)
    df = df.drop(columns=["strategy"], errors="ignore")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.drop(columns=["timestamp"]))
    df_scaled = pd.DataFrame(scaled, columns=[c for c in df.columns if c != "timestamp"])
    df_scaled["timestamp"] = df["timestamp"].values

    features = df_scaled.to_dict(orient="records")

    # ✅ STEP1: class_ranges 동적 계산 → prediction_log.csv → 실패 시 features 기반
    try:
        log_df = pd.read_csv("/persistent/prediction_log.csv", encoding="utf-8-sig")
        gains = pd.to_numeric(log_df["return"], errors="coerce").dropna().values
        gains = gains[np.isfinite(gains)]
        if len(gains) < NUM_CLASSES:
            raise Exception("prediction_log.csv 데이터 부족")
        percentiles = np.percentile(gains, np.linspace(0, 100, NUM_CLASSES+1))
        class_ranges = list(zip(percentiles[:-1], percentiles[1:]))
    except Exception as e:
        print(f"[⚠️ prediction_log.csv 실패 → features 기반 계산 시도] {e}")
        gains = []
        for i in range(window, len(features) - 3):
            base = features[i]
            entry_price = float(base.get("close", 0.0))
            future = features[i+1:]
            if entry_price <= 0 or len(future) < 1:
                continue
            max_future_price = max(f.get("high", f.get("close", entry_price)) for f in future)
            gain = float((max_future_price - entry_price) / (entry_price + 1e-6))
            if np.isfinite(gain):
                gains.append(gain)
        if len(gains) < NUM_CLASSES:
            print(f"[⚠️ features gain 계산도 부족 → 기본값 사용]")
            class_ranges = [
                (-1.00, -0.60), (-0.60, -0.30), (-0.30, -0.20), (-0.20, -0.15),
                (-0.15, -0.10), (-0.10, -0.07), (-0.07, -0.05), (-0.05, -0.03),
                (-0.03, -0.01), (-0.01, 0.01), (0.01, 0.03), (0.03, 0.05),
                (0.05, 0.07), (0.07, 0.10), (0.10, 0.15), (0.15, 0.20),
                (0.20, 0.30), (0.30, 0.60), (0.60, 1.00), (1.00, 2.00), (2.00, 5.00)
            ]
        else:
            percentiles = np.percentile(gains, np.linspace(0, 100, NUM_CLASSES+1))
            class_ranges = list(zip(percentiles[:-1], percentiles[1:]))

    strategy_minutes = {"단기": 240, "중기": 1440, "장기": 10080}
    lookahead_minutes = strategy_minutes.get(strategy, 1440)

    for i in range(window, len(features) - 3):
        try:
            seq = features[i - window:i]
            base = features[i]
            entry_time = pd.to_datetime(base.get("timestamp"), errors="coerce")
            entry_price = float(base.get("close", 0.0))

            if pd.isnull(entry_time) or entry_price <= 0:
                continue

            future = [
                f for f in features[i + 1:]
                if "timestamp" in f and pd.to_datetime(f["timestamp"], errors="coerce") - entry_time <= pd.Timedelta(minutes=lookahead_minutes)
            ]

            if len(seq) != window or len(future) < 1:
                continue

            max_future_price = max(f.get("high", f.get("close", entry_price)) for f in future)
            gain = float((max_future_price - entry_price) / (entry_price + 1e-6))
            if pd.isnull(gain) or not np.isfinite(gain):
                gain = 0.0

            cls = next((j for j, (low, high) in enumerate(class_ranges) if low <= gain < high), NUM_CLASSES-1)
            cls = int(cls)

            if cls >= NUM_CLASSES:
                print(f"[⚠️ 라벨 보정] cls {cls} → NUM_CLASSES-1 {NUM_CLASSES-1}")
                cls = NUM_CLASSES - 1

            sample = [[float(r.get(c, 0.0)) for c in columns] for r in seq]

            if input_size:
                for j in range(len(sample)):
                    row = sample[j]
                    if len(row) < input_size:
                        row.extend([0.0] * (input_size - len(row)))
                    elif len(row) > input_size:
                        sample[j] = row[:input_size]

            X.append(sample)
            y.append(cls)

        except Exception as e:
            print(f"[예외 발생] ❌ {e} → i={i}")
            continue

    if not y:
        msg = "[⚠️ 경고] 생성된 라벨 없음"
        print(msg)
        raise Exception(msg)
    else:
        labels, counts = np.unique(y, return_counts=True)
        print(f"[📊 클래스 분포] → {dict(zip(labels, counts))}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def get_kline_by_strategy(symbol: str, strategy: str):
    from predict import failed_result
    import os

    global _kline_cache
    cache_key = f"{symbol}-{strategy}"
    if cache_key in _kline_cache:
        print(f"[캐시 사용] {cache_key}")
        return _kline_cache[cache_key]

    config = STRATEGY_CONFIG.get(strategy)
    if config is None:
        print(f"[❌ 실패] {symbol}-{strategy}: 전략 설정 없음")
        failed_result(symbol, strategy, reason="전략 설정 없음")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = get_kline(symbol, interval=config["interval"], limit=config["limit"])
    if df is None or not isinstance(df, pd.DataFrame):
        print(f"[❌ 실패] {symbol}-{strategy}: get_kline() → None 반환 또는 형식 오류")
        failed_result(symbol, strategy, reason="get_kline 반환 오류")

        # ✅ 수정 추가: API 미수신 심볼 목록 로깅
        try:
            log_path = "/persistent/logs/api_missing_symbols.txt"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{symbol}-{strategy}\n")
            print(f"[📄 기록] API 미수신 심볼 → {symbol}-{strategy} 기록됨")
        except Exception as e:
            print(f"[⚠️ 로깅 실패] API 미수신 심볼 기록 실패: {e}")

        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    required_cols = ["open", "high", "low", "close", "volume", "timestamp"]
    missing = [col for col in required_cols if col not in df.columns]
    nan_cols = [col for col in required_cols if col in df.columns and df[col].isnull().any()]

    if missing:
        print(f"[⚠️ 경고] {symbol}-{strategy}: 필수 컬럼 누락 → {missing}")
        failed_result(symbol, strategy, reason=f"필수컬럼누락:{missing}")
        return pd.DataFrame(columns=required_cols)

    if nan_cols:
        print(f"[⚠️ 경고] {symbol}-{strategy}: NaN 존재 → {nan_cols}")
        failed_result(symbol, strategy, reason=f"NaN존재:{nan_cols}")
        return pd.DataFrame(columns=required_cols)

    if len(df) < 5:
        print(f"[⚠️ 경고] {symbol}-{strategy}: 데이터 row 부족 ({len(df)} rows)")
        failed_result(symbol, strategy, reason="row 부족")
        return pd.DataFrame(columns=required_cols)

    print(f"[✅ 성공] {symbol}-{strategy}: 데이터 {len(df)}개 확보")
    _kline_cache[cache_key] = df
    return df



def get_kline(symbol: str, interval: str = "60", limit: int = 300, max_retry: int = 3) -> pd.DataFrame:
    import time

    for attempt in range(max_retry):
        try:
            url = f"{BASE_URL}/v5/market/kline"
            params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()

            if "result" not in data or "list" not in data["result"]:
                print(f"[경고] get_kline() → 데이터 응답 구조 이상: {symbol}, 재시도 {attempt+1}/{max_retry}")
                time.sleep(1)
                continue

            raw = data["result"]["list"]
            if not raw or len(raw[0]) < 6:
                print(f"[경고] get_kline() → 필수 필드 누락: {symbol}, 재시도 {attempt+1}/{max_retry}")
                time.sleep(1)
                continue

            df = pd.DataFrame(raw, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")

            essential = ["open", "high", "low", "close", "volume"]
            df.dropna(subset=essential, inplace=True)
            if df.empty:
                print(f"[경고] get_kline() → 필수값 결측: {symbol}, 재시도 {attempt+1}/{max_retry}")
                time.sleep(1)
                continue

            if "high" not in df.columns or df["high"].isnull().all() or (df["high"] == 0).all():
                print(f"[치명] get_kline() → 'high' 값 전부 비정상: {symbol}, 재시도 {attempt+1}/{max_retry}")
                time.sleep(1)
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["datetime"] = df["timestamp"]

            return df

        except Exception as e:
            print(f"[에러] get_kline({symbol}) 실패 → {e}, 재시도 {attempt+1}/{max_retry}")
            time.sleep(1)

    print(f"[❌ 실패] get_kline() 최종 실패: {symbol}")
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


_feature_cache = {}

def compute_features(symbol: str, df: pd.DataFrame, strategy: str, required_features: list = None, fallback_input_size: int = None) -> pd.DataFrame:
    from predict import failed_result
    from config import FEATURE_INPUT_SIZE  # ✅ FEATURE_INPUT_SIZE 상수 import
    import ta
    global _feature_cache
    cache_key = f"{symbol}-{strategy}"

    if cache_key in _feature_cache:
        print(f"[캐시 사용] {cache_key} 피처")
        return _feature_cache[cache_key]

    if df is None or df.empty:
        print(f"[❌ compute_features 실패] 입력 DataFrame empty")
        failed_result(symbol, strategy, reason="입력DataFrame empty")
        return pd.DataFrame(columns=["timestamp", "strategy", "close", "high"])

    df = df.copy()

    if "datetime" in df.columns:
        df["timestamp"] = df["datetime"]
    elif "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime("now")

    df["strategy"] = strategy

    try:
        base_cols = ["open", "high", "low", "close", "volume"]
        df = df[["timestamp", "strategy"] + base_cols]

        # ✅ 기존 feature engineering 유지
        df["ma20"] = df["close"].rolling(window=20, min_periods=1).mean()
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-6)
        df["rsi"] = 100 - (100 / (1 + rs))
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["bollinger"] = df["close"].rolling(window=20, min_periods=1).std()
        df["volatility"] = df["high"] - df["low"]
        df["trend_score"] = (df["close"] > df["ma20"]).astype(int)

        # ✅ 고급 기술적 지표 추가
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        df["roc"] = df["close"].pct_change(periods=10)
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14, fillna=True)
        df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"], window=20, fillna=True)
        df["mfi"] = ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"], window=14, fillna=True)
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"], fillna=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        feature_cols = [c for c in df.columns if c not in ["timestamp", "strategy"]]
        print(f"[info] compute_features 생성 feature 개수: {len(feature_cols)} → {feature_cols}")

        # ✅ 수정: FEATURE_INPUT_SIZE 기반 padding 적용
        if len(feature_cols) < FEATURE_INPUT_SIZE:
            pad_cols = []
            for i in range(len(feature_cols), FEATURE_INPUT_SIZE):
                pad_col = f"pad_{i}"
                df[pad_col] = 0.0
                pad_cols.append(pad_col)
            feature_cols += pad_cols
            print(f"[info] feature padding 적용: {pad_cols}")

        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

    except Exception as e:
        print(f"[❌ compute_features 예외] feature 계산 실패 → {e}")
        failed_result(symbol, strategy, reason=f"feature 계산 실패: {e}")
        return pd.DataFrame(columns=["timestamp", "strategy", "close", "high"])

    required_cols = ["timestamp", "close", "high"]
    missing_cols = [col for col in required_cols if col not in df.columns or df[col].isnull().any()]
    if missing_cols or df.empty:
        print(f"[❌ compute_features 실패] 필수 컬럼 누락 또는 NaN 존재: {missing_cols}, rows={len(df)}")
        failed_result(symbol, strategy, reason=f"필수컬럼누락 또는 NaN: {missing_cols}")
        return pd.DataFrame(columns=required_cols + ["strategy"])

    if len(df) < 5:
        print(f"[❌ compute_features 실패] 데이터 row 부족 ({len(df)} rows)")
        failed_result(symbol, strategy, reason="row 부족")
        return pd.DataFrame(columns=required_cols + ["strategy"])

    print(f"[✅ 완료] {symbol}-{strategy}: 피처 {df.shape[0]}개 생성")
    _feature_cache[cache_key] = df
    return df



# data/utils.py 맨 아래에 추가

SYMBOL_GROUPS = [SYMBOLS[i:i+5] for i in range(0, len(SYMBOLS), 5)]
