# ✅ Render 캐시 강제 무효화용 주석 — 절대 삭제하지 마
_kline_cache = {}

import os
import time
import json
import requests
import pandas as pd
import numpy as np
import pytz
from sklearn.preprocessing import MinMaxScaler

# =========================
# 기본 상수/전역
# =========================
BASE_URL = "https://api.bybit.com"
BINANCE_BASE_URL = "https://fapi.binance.com"  # Binance Futures (USDT-M)
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

# ✅ 고정 순서 유지하며 5개씩 묶어 SYMBOL_GROUPS 구성
SYMBOL_GROUPS = [SYMBOLS[i:i + 5] for i in range(0, len(SYMBOLS), 5)]

STRATEGY_CONFIG = {
    "단기": {"interval": "240", "limit": 1000},   # 4시간봉 (240분)
    "중기": {"interval": "D",   "limit": 500},    # 1일봉
    "장기": {"interval": "D",   "limit": 500}     # 1일봉(장기)
}

# 거래소별 심볼 매핑
SYMBOL_MAP = {
    "bybit": {s: s for s in SYMBOLS},
    "binance": {s: s for s in SYMBOLS}
}

# =========================
# 캐시 매니저 (이 파일 내부 사용)
# =========================
class CacheManager:
    _cache = {}
    _ttl = {}

    @classmethod
    def get(cls, key, ttl_sec=None):
        now = time.time()
        if key in cls._cache:
            if ttl_sec is None or now - cls._ttl.get(key, 0) < ttl_sec:
                print(f"[캐시 HIT] {key}")
                return cls._cache[key]
            else:
                print(f"[캐시 EXPIRED] {key}")
                cls.delete(key)
        return None

    @classmethod
    def set(cls, key, value):
        cls._cache[key] = value
        cls._ttl[key] = time.time()
        print(f"[캐시 SET] {key}")

    @classmethod
    def delete(cls, key):
        if key in cls._cache:
            del cls._cache[key]
            del cls._ttl[key]
            print(f"[캐시 DELETE] {key}")

    @classmethod
    def clear(cls):
        cls._cache.clear()
        cls._ttl.clear()
        print("[캐시 CLEAR ALL]")

# =========================
# 실패 로깅(순환 의존 제거용 경량 헬퍼)
# =========================
def safe_failed_result(symbol, strategy, reason=""):
    try:
        from failure_db import insert_failure_record  # 순환 없음
        payload = {
            "symbol": symbol or "UNKNOWN",
            "strategy": strategy or "UNKNOWN",
            "model": "utils",
            "reason": reason,
            "timestamp": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_class": -1,
            "label": -1
        }
        insert_failure_record(payload, feature_hash="utils_error", feature_vector=None, label=-1)
    except Exception as e:
        print(f"[⚠️ safe_failed_result 실패] {e}")

# =========================
# 기타 유틸
# =========================
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

# =========================
# 데이터셋 생성
# =========================
def create_dataset(features, window=10, strategy="단기", input_size=None):
    import pandas as pd
    from config import MIN_FEATURES
    from logger import log_prediction
    from collections import Counter
    import random

    X, y = [], []

    def get_symbol_safe():
        if isinstance(features, list) and features and isinstance(features[0], dict) and "symbol" in features[0]:
            return features[0]["symbol"]
        return "UNKNOWN"

    if not isinstance(features, list) or len(features) <= window:
        print(f"[⚠️ 부족] features length={len(features) if isinstance(features, list) else 'Invalid'}, window={window} → dummy 반환")
        dummy_X = np.zeros((1, window, input_size if input_size else MIN_FEATURES), dtype=np.float32)
        dummy_y = np.array([-1], dtype=np.int64)
        log_prediction(symbol=get_symbol_safe(), strategy=strategy, direction="dummy", entry_price=0,
                       target_price=0, model="dummy_model", success=False, reason="입력 feature 부족",
                       rate=0.0, return_value=0.0, volatility=False, source="create_dataset",
                       predicted_class=-1, label=-1)
        return dummy_X, dummy_y

    try:
        df = pd.DataFrame(features)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = df.drop(columns=["strategy"], errors="ignore")

        feature_cols = [c for c in df.columns if c not in ["timestamp"]]
        if not feature_cols:
            raise ValueError("feature_cols 없음")

        # 최소 피처 보장
        if len(feature_cols) < MIN_FEATURES:
            for i in range(len(feature_cols), MIN_FEATURES):
                pad_col = f"pad_{i}"
                df[pad_col] = 0.0
                feature_cols.append(pad_col)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[feature_cols])
        df_scaled = pd.DataFrame(scaled, columns=feature_cols)
        df_scaled["timestamp"] = df["timestamp"].values
        df_scaled["high"] = df["high"] if "high" in df.columns else df["close"]

        if input_size and len(feature_cols) < input_size:
            for i in range(len(feature_cols), input_size):
                pad_col = f"pad_{i}"
                df_scaled[pad_col] = 0.0

        features = df_scaled.to_dict(orient="records")

        strategy_minutes = {"단기": 240, "중기": 1440, "장기": 2880}
        lookahead_minutes = strategy_minutes.get(strategy, 1440)

        valid_gains, samples = [], []

        for i in range(window, len(features)):
            seq = features[i - window:i]
            base = features[i]
            entry_time = pd.to_datetime(base.get("timestamp"), errors="coerce")
            entry_price = float(base.get("close", 0.0))
            if pd.isnull(entry_time) or entry_price <= 0:
                continue

            future = [f for f in features[i + 1:] if pd.to_datetime(f.get("timestamp", None)) - entry_time <= pd.Timedelta(minutes=lookahead_minutes)]
            valid_prices = [f.get("high", f.get("close", entry_price)) for f in future if f.get("high", 0) > 0]
            if len(seq) != window or not valid_prices:
                continue

            max_future_price = max(valid_prices)
            gain = float((max_future_price - entry_price) / (entry_price + 1e-6))
            valid_gains.append(gain)

            row_cols = [c for c in df_scaled.columns if c != "timestamp"]
            sample = [[float(r.get(c, 0.0)) for c in row_cols] for r in seq]
            if input_size:
                for j in range(len(sample)):
                    row = sample[j]
                    if len(row) < input_size:
                        row.extend([0.0] * (input_size - len(row)))
                    elif len(row) > input_size:
                        sample[j] = row[:input_size]
            samples.append((sample, gain))

        if not samples or not valid_gains:
            print("[❌ 수익률 없음] dummy 반환")
            dummy_X = np.random.normal(0, 1, size=(10, window, input_size if input_size else MIN_FEATURES)).astype(np.float32)
            dummy_y = np.random.randint(0, 5, size=(10,))
            return dummy_X, dummy_y

        # 동적 클래스 수 추정 (최대 21, 최소 3)
        min_gain, max_gain = min(valid_gains), max(valid_gains)
        spread = max_gain - min_gain
        est_class = int(spread / 0.01)
        num_classes = max(3, min(21, est_class))

        step = spread / num_classes if num_classes > 0 else 1e-6
        if step == 0:
            step = 1e-6

        X, y = [], []
        for sample, gain in samples:
            cls = min(int((gain - min_gain) / step), num_classes - 1)
            X.append(sample); y.append(cls)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        print(f"[✅ create_dataset 완료] 샘플 수: {len(y)}, X.shape={X.shape}, 동적 클래스 수: {num_classes}")
        return X, y, num_classes  # ⬅ 필요 시 사용

    except Exception as e:
        print(f"[❌ 최상위 예외] create_dataset 실패 → {e}")
        dummy_X = np.random.normal(0, 1, size=(10, window, 8)).astype(np.float32)
        dummy_y = np.random.randint(0, 5, size=(10,))
        return dummy_X, dummy_y, 5

# =========================
# 거래소 수집기
# =========================
def get_kline(symbol: str, interval: str = "60", limit: int = 300, max_retry: int = 2, end_time=None) -> pd.DataFrame:
    real_symbol = SYMBOL_MAP["bybit"].get(symbol, symbol)
    target_rows = int(limit)
    collected_data, total_rows = [], 0

    while total_rows < target_rows:
        success = False
        for attempt in range(max_retry):
            try:
                rows_needed = target_rows - total_rows
                request_limit = min(1000, rows_needed)
                params = {
                    "category": "linear",
                    "symbol": real_symbol,
                    "interval": interval,
                    "limit": request_limit
                }
                if end_time is not None:
                    params["end"] = int(end_time.timestamp() * 1000)

                print(f"[📡 Bybit 요청] {real_symbol}-{interval} | 시도 {attempt+1}/{max_retry} | 요청 수량={request_limit}")
                res = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
                res.raise_for_status()
                data = res.json()

                if "result" not in data or "list" not in data["result"] or not data["result"]["list"]:
                    print(f"[❌ 데이터 없음] {real_symbol} (시도 {attempt+1})")
                    break

                raw = data["result"]["list"]
                if not raw or len(raw[0]) < 6:
                    print(f"[❌ 필드 부족] {real_symbol}")
                    break

                df_chunk = pd.DataFrame(raw, columns=[
                    "timestamp", "open", "high", "low", "close", "volume", "turnover"
                ])
                df_chunk = df_chunk[["timestamp", "open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
                df_chunk.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
                if df_chunk.empty:
                    break

                df_chunk["timestamp"] = pd.to_datetime(df_chunk["timestamp"], unit="ms", errors="coerce")
                df_chunk.dropna(subset=["timestamp"], inplace=True)
                df_chunk["timestamp"] = df_chunk["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
                df_chunk = df_chunk.sort_values("timestamp").reset_index(drop=True)
                df_chunk["datetime"] = df_chunk["timestamp"]

                collected_data.append(df_chunk)
                total_rows += len(df_chunk)
                success = True

                if total_rows >= target_rows:
                    break

                end_time = df_chunk["timestamp"].min() - pd.Timedelta(milliseconds=1)
                time.sleep(0.2)
                break

            except Exception as e:
                print(f"[에러] get_kline({real_symbol}) 실패 → {e}")
                time.sleep(1)
                continue

        if not success:
            break

    if collected_data:
        df = pd.concat(collected_data, ignore_index=True) \
               .drop_duplicates(subset=["timestamp"]) \
               .sort_values("timestamp") \
               .reset_index(drop=True)
        print(f"[📊 수집 완료] {symbol}-{interval} → 총 {len(df)}개 봉 확보")
        return df
    else:
        print(f"[❌ 최종 실패] {symbol}-{interval} → 수집된 봉 없음")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"])

def get_kline_binance(symbol: str, interval: str = "240", limit: int = 300, max_retry: int = 2, end_time=None) -> pd.DataFrame:
    real_symbol = SYMBOL_MAP["binance"].get(symbol, symbol)
    interval_map = {"240": "4h", "D": "1d", "2D": "2d", "60": "1h"}
    binance_interval = interval_map.get(interval, "1h")

    target_rows = int(limit)
    collected_data, total_rows = [], 0

    while total_rows < target_rows:
        success = False
        for attempt in range(max_retry):
            try:
                rows_needed = target_rows - total_rows
                request_limit = min(1000, rows_needed)
                params = {
                    "symbol": real_symbol,
                    "interval": binance_interval,
                    "limit": request_limit
                }
                if end_time is not None:
                    params["endTime"] = int(end_time.timestamp() * 1000)

                print(f"[📡 Binance 요청] {real_symbol}-{interval} | 요청 {request_limit}개 | 시도 {attempt+1}/{max_retry} | end_time={end_time}")
                res = requests.get(f"{BINANCE_BASE_URL}/fapi/v1/klines", params=params, timeout=10)
                res.raise_for_status()
                raw = res.json()
                if not raw:
                    print(f"[❌ Binance 데이터 없음] {real_symbol}-{interval} (시도 {attempt+1})")
                    break

                df_chunk = pd.DataFrame(raw, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "trades", "taker_base_vol", "taker_quote_vol", "ignore"
                ])
                df_chunk = df_chunk[["timestamp", "open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
                df_chunk["timestamp"] = pd.to_datetime(df_chunk["timestamp"], unit="ms", errors="coerce") \
                                          .dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
                df_chunk = df_chunk.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                df_chunk["datetime"] = df_chunk["timestamp"]

                if df_chunk.empty:
                    break

                collected_data.append(df_chunk)
                total_rows += len(df_chunk)
                success = True

                if total_rows >= target_rows:
                    break

                oldest_ts = df_chunk["timestamp"].min()
                end_time = oldest_ts - pd.Timedelta(milliseconds=1)
                time.sleep(0.2)
                break

            except Exception as e:
                print(f"[에러] get_kline_binance({real_symbol}) 실패 → {e}")
                time.sleep(1)
                continue

        if not success:
            break

    if collected_data:
        df = pd.concat(collected_data, ignore_index=True) \
               .drop_duplicates(subset=["timestamp"]) \
               .sort_values("timestamp") \
               .reset_index(drop=True)
        print(f"[📊 Binance 수집 완료] {symbol}-{interval} → 총 {len(df)}개 봉 확보")
        return df
    else:
        print(f"[❌ 최종 실패] {symbol}-{interval} → 수집된 봉 없음")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"])

def get_merged_kline_by_strategy(symbol: str, strategy: str) -> pd.DataFrame:
    config = STRATEGY_CONFIG.get(strategy)
    if not config:
        print(f"[❌ 실패] 전략 설정 없음: {strategy}")
        return pd.DataFrame()

    interval = config["interval"]
    base_limit = int(config["limit"])
    max_total = base_limit

    def fetch_until_target(fetch_func, source_name):
        total_data = []
        end_time = None
        total_count = 0
        max_repeat = 10

        print(f"[⏳ {source_name} 데이터 수집 시작] {symbol}-{strategy} | 목표 {base_limit}개")
        while total_count < max_total and len(total_data) < max_repeat:
            df_chunk = fetch_func(symbol, interval=interval, limit=base_limit, end_time=end_time)
            if df_chunk is None or df_chunk.empty:
                break

            total_data.append(df_chunk)
            total_count += len(df_chunk)

            if len(df_chunk) < base_limit:
                break

            oldest_ts = df_chunk["timestamp"].min()
            end_time = oldest_ts - pd.Timedelta(milliseconds=1)

        df_final = pd.concat(total_data, ignore_index=True) if total_data else pd.DataFrame()
        print(f"[✅ {source_name} 수집 완료] {symbol}-{strategy} → {len(df_final)}개")
        return df_final

    # 1차 Bybit 수집
    df_bybit = fetch_until_target(get_kline, "Bybit")

    # 2차 Binance 보충
    df_binance = pd.DataFrame()
    if len(df_bybit) < base_limit:
        print(f"[⏳ Binance 보충 시작] 부족 {base_limit - len(df_bybit)}개")
        df_binance = fetch_until_target(get_kline_binance, "Binance")

    # 병합
    df_all = pd.concat([df_bybit, df_binance], ignore_index=True)
    if df_all.empty:
        print(f"[⏩ 학습 스킵] {symbol}-{strategy} → 거래소 데이터 전무")
        return pd.DataFrame()

    df_all = df_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df_all.columns:
            df_all[col] = 0.0 if col != "timestamp" else pd.Timestamp.now(tz="Asia/Seoul")

    df_all.attrs["augment_needed"] = len(df_all) < base_limit
    print(f"[🔄 병합 완료] {symbol}-{strategy} → 최종 {len(df_all)}개 (목표 {base_limit}개)")
    if len(df_all) < base_limit:
        print(f"[⚠️ 경고] {symbol}-{strategy} 데이터 부족 ({len(df_all)}/{base_limit})")

    return df_all

# =========================
# 전략별 Kline 수집(캐시포함)
# =========================
def get_kline_by_strategy(symbol: str, strategy: str):
    cache_key = f"{symbol}-{strategy}"
    cached_df = CacheManager.get(cache_key, ttl_sec=600)
    if cached_df is not None:
        print(f"[✅ 캐시 사용] {symbol}-{strategy} → {len(cached_df)}개 봉")
        return cached_df

    try:
        config = STRATEGY_CONFIG.get(strategy, {"limit": 300})
        limit = config.get("limit", 300)
        interval = config.get("interval", "D")

        # 1차: Bybit 반복 수집
        df_bybit = []
        total_bybit = 0
        end_time = None
        print(f"[📡 Bybit 1차 반복 수집 시작] {symbol}-{strategy} (limit={limit})")
        while total_bybit < limit:
            df_chunk = get_kline(symbol, interval=interval, limit=limit, end_time=end_time)
            if df_chunk is None or df_chunk.empty:
                break
            df_bybit.append(df_chunk)
            total_bybit += len(df_chunk)
            end_time = df_chunk["timestamp"].min() - pd.Timedelta(milliseconds=1)
            if len(df_chunk) < limit:
                break

        df_bybit = pd.concat(df_bybit, ignore_index=True) \
            .drop_duplicates(subset=["timestamp"]) \
            .sort_values("timestamp") \
            .reset_index(drop=True) if df_bybit else pd.DataFrame()

        # 2차: Binance 보완 수집
        df_binance = []
        total_binance = 0
        if len(df_bybit) < int(limit * 0.9):
            print(f"[📡 Binance 2차 반복 수집 시작] {symbol}-{strategy} (limit={limit})")
            end_time = None
            while total_binance < limit:
                try:
                    df_chunk = get_kline_binance(symbol, interval=interval, limit=limit, end_time=end_time)
                    if df_chunk is None or df_chunk.empty:
                        break
                    df_binance.append(df_chunk)
                    total_binance += len(df_chunk)
                    end_time = df_chunk["timestamp"].min() - pd.Timedelta(milliseconds=1)
                    if len(df_chunk) < limit:
                        break
                except Exception as be:
                    print(f"[❌ Binance 수집 실패] {symbol}-{strategy} → {be}")
                    break

        df_binance = pd.concat(df_binance, ignore_index=True) \
            .drop_duplicates(subset=["timestamp"]) \
            .sort_values("timestamp") \
            .reset_index(drop=True) if df_binance else pd.DataFrame()

        # 병합
        df_list = [df for df in [df_bybit, df_binance] if not df.empty]
        df = pd.concat(df_list, ignore_index=True) \
            .drop_duplicates(subset=["timestamp"]) \
            .sort_values("timestamp") \
            .reset_index(drop=True) if df_list else pd.DataFrame()

        total_count = len(df)
        if total_count < limit:
            print(f"[⚠️ 수집 수량 부족] {symbol}-{strategy} → 총 {total_count}개 (목표: {limit})")
        else:
            print(f"[✅ 수집 성공] {symbol}-{strategy} → 총 {total_count}개")

        CacheManager.set(cache_key, df)
        return df

    except Exception as e:
        print(f"[❌ 데이터 수집 실패] {symbol}-{strategy} → {e}")
        safe_failed_result(symbol, strategy, reason=str(e))
        return pd.DataFrame()

# =========================
# 프리패치
# =========================
def prefetch_symbol_groups(strategy: str):
    for group in SYMBOL_GROUPS:
        for sym in group:
            try:
                get_kline_by_strategy(sym, strategy)
            except Exception as e:
                print(f"[⚠️ prefetch 실패] {sym}-{strategy}: {e}")

# =========================
# 실시간 티커
# =========================
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

# =========================
# 피처 생성
# =========================
_feature_cache = {}

def compute_features(symbol: str, df: pd.DataFrame, strategy: str, required_features: list = None, fallback_input_size: int = None) -> pd.DataFrame:
    from config import FEATURE_INPUT_SIZE
    import ta

    cache_key = f"{symbol}-{strategy}-features"
    cached_feat = CacheManager.get(cache_key, ttl_sec=600)
    if cached_feat is not None:
        print(f"[캐시 HIT] {cache_key}")
        return cached_feat

    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        print(f"[❌ compute_features 실패] 입력 DataFrame empty or invalid")
        safe_failed_result(symbol, strategy, reason="입력DataFrame empty")
        return pd.DataFrame()

    df = df.copy()
    if "datetime" in df.columns:
        df["timestamp"] = df["datetime"]
    elif "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime("now", utc=True).tz_convert("Asia/Seoul")

    df["strategy"] = strategy  # 로그용
    base_cols = ["open", "high", "low", "close", "volume"]
    for col in base_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df[["timestamp"] + base_cols]

    if len(df) < 20:
        print(f"[⚠️ 피처 실패] {symbol}-{strategy} → row 수 부족: {len(df)}")
        safe_failed_result(symbol, strategy, reason=f"row 부족 {len(df)}")
        return df  # 최소 반환

    try:
        # ✅ 기본 기술지표
        df["ma20"] = df["close"].rolling(window=20, min_periods=1).mean()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-6)
        df["rsi"] = 100 - (100 / (1 + rs))
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["bollinger"] = df["close"].rolling(window=20, min_periods=1).std()
        df["volatility"] = df["high"] - df["low"]
        df["trend_score"] = (df["close"] > df["ma20"]).astype(int)
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        df["roc"] = df["close"].pct_change(periods=10)
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14, fillna=True)
        df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"], window=20, fillna=True)
        df["mfi"] = ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"], window=14, fillna=True)
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"], fillna=True)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14, fillna=True)
        df["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=14, fillna=True)
        df["stoch_k"] = ta.momentum.stoch(df["high"], df["low"], df["close"], fillna=True)
        df["stoch_d"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"], fillna=True)
        df["vwap"] = (df["volume"] * df["close"]).cumsum() / (df["volume"].cumsum() + 1e-6)

        # ✅ 스케일링 및 패딩
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        feature_cols = [c for c in df.columns if c != "timestamp"]
        if len(feature_cols) < FEATURE_INPUT_SIZE:
            for i in range(len(feature_cols), FEATURE_INPUT_SIZE):
                pad_col = f"pad_{i}"
                df[pad_col] = 0.0
                feature_cols.append(pad_col)

        df[feature_cols] = MinMaxScaler().fit_transform(df[feature_cols])

    except Exception as e:
        print(f"[❌ compute_features 실패] feature 계산 예외 → {e}")
        safe_failed_result(symbol, strategy, reason=f"feature 계산 실패: {e}")
        return df  # 최소 구조라도 반환

    if df.empty or df.isnull().values.any():
        print(f"[❌ compute_features 실패] 결과 DataFrame 문제 → 빈 df 또는 NaN 존재")
        safe_failed_result(symbol, strategy, reason="최종 결과 DataFrame 오류")
        return df

    print(f"[✅ 완료] {symbol}-{strategy}: 피처 {df.shape[0]}개 생성")
    print(f"[🔍 feature 상태] {symbol}-{strategy} → shape: {df.shape}, NaN: {df.isnull().values.any()}, 컬럼수: {len(df.columns)}")
    CacheManager.set(cache_key, df)
    return df
