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

def main(strategy, symbols=None):
    from telegram_bot import send_message

    print(f"\n📋 [예측 시작] 전략: {strategy} | 시각: {now_kst().strftime('%Y-%m-%d %H:%M:%S')}")
    total, success, failed, skipped = 0, 0, 0, 0
    failed_cases = []

    target_symbols = symbols if symbols else SYMBOLS

    for symbol in target_symbols:
        if not model_exists(symbol, strategy):
            skipped += 1
            print(f"⏭️ SKIP: {symbol}-{strategy} → 모델 없음")
            continue

        total += 1
        try:
            results = predict(symbol, strategy)
            if not isinstance(results, list) or len(results) == 0:
                failed += 1
                failed_cases.append((symbol, strategy, "예측 결과 없음"))
                print(f"❌ 실패: {symbol}-{strategy} → 예측 결과 없음")
                continue

            all_failed = True
            for result in results:
                if result.get("success", False):
                    all_failed = False
                    direction = result.get("direction", "?")
                    rate = result.get("rate", 0)
                    print(f"✅ 성공: {symbol}-{strategy}-{result['model']} → {direction} | 수익률: {rate:.2%}")
                else:
                    reason = result.get("reason", "이유 없음")
                    print(f"❌ 실패: {symbol}-{strategy}-{result.get('model', '?')} → {reason}")

            if all_failed:
                failed += 1
                failed_cases.append((symbol, strategy, "모든 모델 실패"))
            else:
                success += 1

        except Exception as e:
            failed += 1
            failed_cases.append((symbol, strategy, f"예외: {e}"))
            print(f"⚠️ 예외 발생: {symbol}-{strategy} → {e}")
            traceback.print_exc()

    print("\n📌 === 예측 요약 ===")
    print(f"▶️ 총 시도: {total}")
    print(f"✅ 성공: {success}")
    print(f"❌ 실패: {failed}")
    print(f"⏭️ 모델 없음 SKIP: {skipped}")

    if failed_cases:
        print("\n🧨 실패 목록:")
        for symbol, strategy, reason in failed_cases:
            print(f"- {symbol}-{strategy} → {reason}")

    # ✅ 예측 완료 후 메시지 전송
    send_message(f"📡 전략 {strategy} 예측 완료: 성공 {success} / 실패 {failed}")

# 기존 test_all_predictions 유지 (선택 실행 가능)
def test_all_predictions():
    print(f"\n📋 [예측 점검 시작] {now_kst().strftime('%Y-%m-%d %H:%M:%S')}")
    for strategy in STRATEGIES:
        main(strategy)

if __name__ == "__main__":
    test_all_predictions()
