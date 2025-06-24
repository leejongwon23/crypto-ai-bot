import pandas as pd
import time
import traceback
import datetime
import pytz

from data.utils import SYMBOLS, get_kline_by_strategy
from recommend import run_prediction
from logger import get_model_success_rate, log_audit_prediction
from logger import log_audit_prediction as log_audit
from predict import get_recent_class_frequencies

# ✅ 반드시 필요 (predict.py에서 import함)
def class_to_expected_return(cls):
    centers = [-0.125, -0.085, -0.06, -0.04, -0.02, -0.01, -0.0025, -0.0005,
               0.0005, 0.0025, 0.01, 0.02, 0.04, 0.06, 0.085, 0.125]
    return centers[cls] if 0 <= cls < len(centers) else 0.0


# ❌ 아래 두 함수는 삭제해야 predict.py의 실제 기능이 작동됨
# def get_recent_class_frequencies(): return {}
# def adjust_probs_with_diversity(pred_probs): return pred_probs

last_trigger_time = {}
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))
TRIGGER_COOLDOWN = {"단기": 3600, "중기": 10800, "장기": 21600}
MODEL_TYPES = ["lstm", "cnn_lstm", "transformer"]

def check_pre_burst_conditions(df, strategy):
    try:
        vol_increasing = df['volume'].iloc[-3] < df['volume'].iloc[-2] < df['volume'].iloc[-1]
        price_range = df['close'].iloc[-6:]
        stable_price = (price_range.max() - price_range.min()) / price_range.mean() < 0.005
        ema_5 = df['close'].ewm(span=5).mean().iloc[-1]
        ema_15 = df['close'].ewm(span=15).mean().iloc[-1]
        ema_60 = df['close'].ewm(span=60).mean().iloc[-1]
        ema_pack = max(ema_5, ema_15, ema_60) - min(ema_5, ema_15, ema_60)
        ema_compressed = ema_pack / df['close'].iloc[-1] < 0.003
        bb_std = df['close'].rolling(window=20).std()
        expanding_band = bb_std.iloc[-2] < bb_std.iloc[-1] and bb_std.iloc[-1] > 0.002

        if strategy == "단기":
            return sum([vol_increasing, stable_price, ema_compressed, expanding_band]) >= 2
        elif strategy == "중기":
            return sum([stable_price, ema_compressed, expanding_band]) >= 2
        elif strategy == "장기":
            return sum([ema_compressed, expanding_band]) >= 1
        else:
            return False
    except Exception as e:
        print(f"[조건 점검 오류] {e}")
        traceback.print_exc()
        return False

def check_model_quality(symbol, strategy):
    try:
        for m in MODEL_TYPES:
            past_success_rate = get_model_success_rate(symbol, strategy, m)
            if past_success_rate >= 0.6:
                return True
        return False
    except Exception as e:
        print(f"[성공률 확인 실패] {symbol}-{strategy}: {e}")
        return False

def run():
    print(f"[트리거 실행] 전조 패턴 감지 시작: {now_kst().isoformat()}")
    for symbol in SYMBOLS:
        for strategy in ["단기", "중기", "장기"]:
            try:
                key = f"{symbol}_{strategy}"
                now = time.time()
                cooldown = TRIGGER_COOLDOWN.get(strategy, 3600)
                if now - last_trigger_time.get(key, 0) < cooldown:
                    continue

                df = get_kline_by_strategy(symbol, strategy)
                if df is None or len(df) < 60:
                    continue

                if not check_model_quality(symbol, strategy):
                    continue

                if check_pre_burst_conditions(df, strategy):
                    print(f"[트리거 포착] {symbol} - {strategy} 예측 실행")
                    try:
                        run_prediction(symbol, strategy, source="변동성")
                        last_trigger_time[key] = now
                        log_audit(symbol, strategy, "트리거예측", "조건 만족으로 실행")
                    except Exception as inner:
                        print(f"[예측 실행 실패] {symbol}-{strategy}: {inner}")
                        log_audit(symbol, strategy, "트리거예측오류", f"예측실행실패: {inner}")
            except Exception as e:
                print(f"[트리거 오류] {symbol} {strategy}: {e}")
                log_audit(symbol, strategy or "알수없음", "트리거오류", str(e))
