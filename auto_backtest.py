# auto_backtest.py — 백테스트 자동 실행 스크립트

import os
import pandas as pd
from recommend import recommend_strategy

# 📁 백테스트 대상 심볼 리스트
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

def run_backtest():
    accuracy_count = 0
    total_count = 0

    for symbol in symbols:
        file_path = f"data/{symbol}_test.csv"
        if not os.path.exists(file_path):
            print(f"⚠️ 파일 없음: {file_path}")
            continue

        df = pd.read_csv(file_path)
        result = recommend_strategy(df)

        if result:
            trend, confidence = result
            future_price = df["close"].iloc[-1]
            now_price = df["close"].iloc[-30]
            real_trend = "📈 상승" if future_price > now_price else "📉 하락"

            if trend == real_trend:
                accuracy_count += 1
            total_count += 1

    if total_count == 0:
        print("❌ 테스트 가능한 데이터 없음")
        return

    accuracy = (accuracy_count / total_count) * 100
    print(f"✅ 백테스트 결과: {accuracy:.2f}% 정확도 ({accuracy_count}/{total_count})")

if __name__ == "__main__":
    run_backtest()
