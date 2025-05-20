import pandas as pd
from data.utils import SYMBOLS, get_kline_by_strategy
from recommend import run_prediction
import traceback
import datetime
import pytz

last_trigger_time = {}

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def check_pre_burst_conditions(df):
    try:
        vol_increasing = (
            df['volume'].iloc[-3] < df['volume'].iloc[-2] < df['volume'].iloc[-1]
        )
        price_range = df['close'].iloc[-6:]
        stable_price = (price_range.max() - price_range.min()) / price_range.mean() < 0.005
        if 'strength' in df.columns:
            avg_strength = df['strength'].iloc[-10:-1].mean()
            sudden_strength = df['strength'].iloc[-1] > avg_strength * 2
        else:
            sudden_strength = False
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
            sudden_strength,
            ema_compressed,
            expanding_band
        ])
        return satisfied >= 2
    except Exception:
        traceback.print_exc()
        return False

def run():
    print(f"[트리거 실행] 전조 패턴 감지 시작: {now_kst().isoformat()}")
    for symbol in SYMBOLS:
        for strategy in ['단기', '중기', '장기']:   # 전략별 반복 추가
            try:
                df = get_kline_by_strategy(symbol, strategy)
                if df is None or len(df) < 60:
                    continue

                if check_pre_burst_conditions(df):
                    print(f"[트리거 포착] {symbol} - {strategy} 예측 실행")
                    run_prediction(symbol, strategy)   # 전략명 같이 넘김

            except Exception as e:
                print(f"[트리거 오류] {symbol} {strategy}: {e}")
