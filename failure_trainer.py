from failure_db import load_failure_samples
from train import train_one_model

def run_failure_training():
    """
    ✅ 실패했던 예측 샘플들로 다시 학습을 시도
    """
    failure_data = load_failure_samples()

    if not failure_data:
        print("✅ 실패 샘플 없음 → 실패학습 생략")
        return

    grouped = {}
    for item in failure_data:
        key = (item["symbol"], item["strategy"])
        grouped.setdefault(key, []).append(item)

    for (symbol, strategy), samples in grouped.items():
        print(f"\n🚨 실패 학습 시작: {symbol}-{strategy}, 샘플 수: {len(samples)}")

        try:
            # ✅ 실패학습 유도용 로그 + 인자 주석
            print(f"[⚠️ 실패기반 학습 유도 중] {symbol}-{strategy}")
            train_one_model(symbol, strategy)  # 내부에서 실패샘플 포함 학습되도록 구성되어 있음
        except Exception as e:
            print(f"[❌ 실패 학습 예외] {symbol}-{strategy} → {e}")
