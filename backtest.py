import os
import pandas as pd
from recommend import recommend_strategy

symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
accuracy_count = 0
total_count = 0

for symbol in symbols:
    file_path = f"data/{symbol}_test.csv"
    if not os.path.exists(file_path):
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

accuracy = (accuracy_count / total_count) * 100 if total_count else 0
print(f"✅ 백테스트 정확도: {accuracy:.2f}% ({accuracy_count}/{total_count})")
