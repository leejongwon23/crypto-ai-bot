import os
from data.utils import SYMBOLS
from train import train_model
import time

PERSIST_DIR = "/persistent"
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
REQUIRED_MODELS = ["lstm", "cnn_lstm", "transformer"]

def model_exists(symbol, strategy, model_type):
    filename = f"{symbol}_{strategy}_{model_type}.pt"
    return os.path.exists(os.path.join(MODEL_DIR, filename))

def check_and_train_models():
    print("🔍 모델 존재 여부 점검 시작...")
    missing = []

    for symbol in SYMBOLS:
        for strategy in ["단기", "중기", "장기"]:
            for model_type in REQUIRED_MODELS:
                if not model_exists(symbol, strategy, model_type):
                    missing.append((symbol, strategy))

    # 중복 제거 (같은 조합 여러 모델이 누락됐을 수 있음)
    missing = list(set(missing))

    if not missing:
        print("✅ 모든 모델이 정상적으로 존재합니다.")
        return

    print(f"⚠️ 누락된 모델 조합: {len(missing)}개 → 자동 학습 시작")

    for symbol, strategy in missing:
        try:
            print(f"⏳ {symbol}-{strategy} 모델 학습 시작")
            train_model(symbol, strategy)
            time.sleep(1)  # 학습 사이 간격 약간 줌 (안정성)
        except Exception as e:
            print(f"[오류] {symbol}-{strategy} 학습 실패: {e}")

    print("✅ 누락 모델 자동 학습 완료")
