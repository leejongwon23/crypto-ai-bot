import os
from predict import predict
from data.utils import SYMBOLS
from model_weight_loader import model_exists
import datetime
import pytz
import traceback

STRATEGIES = ["단기", "중기", "장기"]

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def test_all_predictions():
    print(f"\n📋 [예측 점검 시작] {now_kst().strftime('%Y-%m-%d %H:%M:%S')}")
    total, success, failed, skipped = 0, 0, 0, 0
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
                if not isinstance(result, dict):
                    failed += 1
                    failed_cases.append((symbol, strategy, "None 또는 dict 아님"))
                    print(f"❌ 실패: {symbol}-{strategy} → 반환 없음 또는 비정상")
                    continue

                if not result.get("success", False):
                    reason = result.get("reason", "이유 없음")
                    failed += 1
                    failed_cases.append((symbol, strategy, reason))
                    print(f"❌ 실패: {symbol}-{strategy} → {reason}")
                    continue

                direction = result.get("direction", "?")
                rate = result.get("rate", 0)
                print(f"✅ 성공: {symbol}-{strategy} → {direction} | 수익률: {rate:.2%}")
                success += 1

            except Exception as e:
                failed += 1
                failed_cases.append((symbol, strategy, f"예외: {e}"))
                print(f"⚠️ 예외 발생: {symbol}-{strategy} → {e}")
                traceback.print_exc()

    print("\n📌 === 예측 점검 요약 ===")
    print(f"▶️ 총 시도: {total}")
    print(f"✅ 성공: {success}")
    print(f"❌ 실패: {failed}")
    print(f"⏭️ 모델 없음 SKIP: {skipped}")
    if failed_cases:
        print("\n🧨 실패 목록:")
        for symbol, strategy, reason in failed_cases:
            print(f"- {symbol}-{strategy} → {reason}")

if __name__ == "__main__":
    test_all_predictions()
