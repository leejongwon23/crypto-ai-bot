import os
from predict import predict
from data.utils import SYMBOLS
from model_weight_loader import model_exists
import datetime
import pytz

STRATEGIES = ["단기", "중기", "장기"]

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def test_all_predictions():
    print(f"\n📋 [예측 점검 시작] {now_kst().strftime('%Y-%m-%d %H:%M:%S')}")

    total = 0
    success = 0
    failed = 0
    skipped = 0
    failed_cases = []

    for strategy in STRATEGIES:
        for symbol in SYMBOLS:
            if not model_exists(symbol, strategy):
                skipped += 1
                print(f"⏭️ SKIP: {symbol}-{strategy} → 모델 없음")
                continue

            total += 1
            try:
                result = predict(symbol, strategy)
                if result is None:
                    failed += 1
                    failed_cases.append((symbol, strategy))
                    print(f"❌ 실패: {symbol}-{strategy} → None 반환")
                else:
                    success += 1
                    direction = result.get("direction", "?")
                    conf = result.get("confidence", 0)
                    rate = result.get("rate", 0)
                    print(f"✅ 성공: {symbol}-{strategy} → {direction} | 신뢰도: {conf:.2f} / 수익률: {rate:.2%}")
            except Exception as e:
                failed += 1
                failed_cases.append((symbol, strategy))
                print(f"⚠️ 예외 발생: {symbol}-{strategy} → {e}")

    print("\n📌 === 예측 점검 요약 ===")
    print(f"▶️ 총 시도: {total}")
    print(f"✅ 성공: {success}")
    print(f"❌ 실패: {failed}")
    print(f"⏭️ 모델 없음 SKIP: {skipped}")
    if failed_cases:
        print("🧨 실패 목록:")
        for symbol, strategy in failed_cases:
            print(f"- {symbol}-{strategy}")

if __name__ == "__main__":
    test_all_predictions()
