# model_checker.py (FINAL)

import os
import time
from glob import glob

from data.utils import SYMBOLS
from train import train_models  # ✅ 실제 존재하는 API로 교체

PERSIST_DIR = "/persistent"
MODEL_DIR = os.path.join(PERSIST_DIR, "models")
REQUIRED_MODELS = ["lstm", "cnn_lstm", "transformer"]


def model_exists(symbol: str, strategy: str, model_type: str) -> bool:
    """
    학습 산출물 실제 규칙:
      {symbol}_{strategy}_{model_type}_group{gid}_cls{n}.pt
    메타(json)까지 존재하는지 확인하여 가용 모델 판단.
    """
    # 예: BTCUSDT_단기_lstm_group0_cls5.pt
    pattern = os.path.join(
        MODEL_DIR, f"{symbol}_{strategy}_{model_type}_group*_cls*.pt"
    )
    for pt_path in glob(pattern):
        meta_path = pt_path.replace(".pt", ".meta.json")
        if os.path.exists(meta_path):
            return True
    return False


def check_and_train_models():
    """
    - 모든 심볼 × 3전략 × 3모델타입 점검
    - 누락된 (심볼, 전략)이 하나라도 있으면 해당 심볼 전체를 train_models([symbol])로 학습
      (train.py의 공개 API에 맞춘 안전 호출)
    """
    print("🔍 모델 존재 여부 점검 시작...")
    missing_pairs = set()

    for symbol in SYMBOLS:
        for strategy in ["단기", "중기", "장기"]:
            # 세 모델 중 하나라도 없으면 학습 큐에 추가
            for model_type in REQUIRED_MODELS:
                if not model_exists(symbol, strategy, model_type):
                    missing_pairs.add((symbol, strategy))
                    break  # 이 전략은 학습 대상 확정, 더 볼 필요 없음

    if not missing_pairs:
        print("✅ 모든 모델이 정상적으로 존재합니다.")
        return

    print(f"⚠️ 누락된 (심볼, 전략) 조합: {len(missing_pairs)}개 → 자동 학습 시작")

    # 심볼 단위로 묶어서 학습(같은 심볼 여러 전략이 빠졌을 수 있음)
    symbols_to_train = sorted({s for (s, _) in missing_pairs})
    for symbol in symbols_to_train:
        try:
            print(f"⏳ {symbol} 전체 전략 학습 시작 (train_models([{symbol!r}]))")
            train_models([symbol])  # ✅ train.py의 공개 함수
            time.sleep(1)  # 안정성 간격
        except Exception as e:
            print(f"[오류] {symbol} 학습 실패: {e}")

    print("✅ 누락 모델 자동 학습 완료")


if __name__ == "__main__":
    check_and_train_models()
