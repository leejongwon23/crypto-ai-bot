# accuracy_logger.py (왕1 보완 기능: 예측 결과 정확도 추적 및 CSV 저장)

import csv
import os
from datetime import datetime

def log_prediction(symbol, timeframe, true_label, predicted_prob):
    """
    예측 결과 로깅 함수
    - symbol: 코인 심볼
    - timeframe: 'short' | 'mid' | 'long'
    - true_label: 실제 정답 (0 또는 1)
    - predicted_prob: 모델이 예측한 확률 (0~1)
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "prediction_accuracy.csv")

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    predicted_label = 1 if predicted_prob >= 0.5 else 0
    correct = 1 if predicted_label == true_label else 0

    row = [now, symbol, timeframe, true_label, predicted_label, predicted_prob, correct]

    header = ["timestamp", "symbol", "timeframe", "true_label", "predicted_label", "predicted_prob", "correct"]
    write_header = not os.path.exists(log_path)

    with open(log_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"✅ 로깅 완료: {symbol} [{timeframe}] 정답={true_label}, 예측={predicted_label}, 확률={predicted_prob:.2f}")
