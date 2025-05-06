import pandas as pd
import numpy as np

# 1000개 샘플을 랜덤으로 생성 (close 기준 시계열 흐름을 갖도록 구성)
np.random.seed(42)
base_price = 100
closes = [base_price]
for _ in range(999):
    closes.append(closes[-1] * (1 + np.random.normal(0, 0.005)))  # 약 ±0.5% 변동

data = {
    "timestamp": pd.date_range(start="2024-01-01", periods=1000, freq="H"),
    "open": [round(c * np.random.uniform(0.99, 1.01), 2) for c in closes],
    "high": [round(c * np.random.uniform(1.00, 1.02), 2) for c in closes],
    "low":  [round(c * np.random.uniform(0.98, 1.00), 2) for c in closes],
    "close": [round(c, 2) for c in closes],
    "volume": np.random.uniform(1000, 5000, size=1000)
}

df = pd.DataFrame(data)
df.to_csv("sample_training_data.csv", index=False)
print("✅ sample_training_data.csv 파일 생성 완료!")
