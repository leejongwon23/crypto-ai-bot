import pandas as pd
import time
import traceback
import datetime
import pytz

from data.utils import SYMBOLS, get_kline_by_strategy
from recommend import run_prediction
from logger import get_model_success_rate

# ✅ 최근 트리거 시간 기록
last_trigger_time = {}

# ✅ 시간 계산 (KST 기준)
def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# ✅ 전략별 최소 트리거 간격 (초)
TRIGGER_COOLDOWN = {
    "단기": 3600,    # 1시간
    "중기": 10800,   # 3시간
    "장기": 21600    # 6시간
}

# ✅ 트리거 조건 점검 함수
def check_pre_burst_conditions(df, strategy):
    try:
        vol_increasing = df['volume'].iloc[-3] < df['volume'].iloc[-2] < df['volume'].iloc[-1]
        price_range = df['close'].iloc[-6:]
        stable_price = (price_range.max() - price_range.min()) / price_range.mean() < 0.005
        ema_5 = df['close'].ewm(span=5).mean().iloc[-1]
        ema_15 = df['close'].ewm(span=15).mean().iloc[-1]
        ema_60 = df['close'].ewm(span=60).mean().iloc[-1]
        ma_pack = max(ema_5, ema_15, ema_60) - min(ema_5, ema_15, ema_60)
        ema_compressed = ma_pack / df['close'].iloc[-1] < 0.003
        bb_std = df['close'].rolling(window=20).std()
        expanding_band = bb_std.iloc[-2] < bb_std.iloc[-1] and bb_std.iloc[-1] > 0.002
        satisfied = sum([
            vol_increasing,
            stable_price,
            ema_compressed,
            expanding_band
        ])
        return satisfied >= 2
    except Exception:
        traceback.print_exc()
        return False

# ✅ 전략별 신뢰도 조건 검사
def check_model_quality(symbol, strategy):
    try:
        success_rate = get_model_success_rate(symbol, strategy, "ensemble")
        return success_rate >= 0.6
    except:
        return False

# ✅ 메인 트리거 실행
def run():
    print(f"[트리거 실행] 전조 패턴 감지 시작: {now_kst().isoformat()}")
    for symbol in SYMBOLS:
        for strategy in ['단기', '중기', '장기']:
            try:
                key = f"{symbol}_{strategy}"
                now = time.time()
                cooldown = TRIGGER_COOLDOWN.get(strategy, 3600)
                last_time = last_trigger_time.get(key, 0)
                if now - last_time < cooldown:
                    continue

                df = get_kline_by_strategy(symbol, strategy)
                if df is None or len(df) < 60:
                    continue

                if not check_model_quality(symbol, strategy):
                    continue

                if check_pre_burst_conditions(df, strategy):
                    print(f"[트리거 포착] {symbol} - {strategy} 예측 실행")
                    run_prediction(symbol, strategy)
                    last_trigger_time[key] = now
            except Exception as e:
                print(f"[트리거 오류] {symbol} {strategy}: {e}")
