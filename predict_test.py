import os
from predict import predict
from data.utils import SYMBOLS
import datetime
import pytz

STRATEGIES = ["단기", "중기", "장기"]

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def test_all_predictions():
    print(f"\n[예측 점검 시작] {now_kst().isoformat()}")

    total = 0
    success = 0
    failed = 0
    failed_cases = []

    for strategy in STRATEGIES:
        for symbol in SYMBOLS:
            total += 1
            try:
                result = predict(symbol, strategy)
                if result is None:
                    failed += 1
                    failed_cases.append((symbol, strategy))
                    print(f"[실패] {symbol}-{strategy} → None 반환")
                else:
                    success += 1
                    print(f"[성공] {symbol}-{strategy} → {result['direction']} | conf: {result['confidence']:.2f}, rate: {result['rate']:.2%}")
            except Exception as e:
                failed += 1
                failed_cases.append((symbol, strategy))
                print(f"[ERROR] {symbol}-{strategy} 예측 중 예외 발생 → {e}")

    print("\n=== 예측 점검 요약 ===")
    print(f"총 시도: {total}")
    print(f"성공: {success}")
    print(f"실패: {failed}")
    if failed_cases:
        print("실패 목록:")
        for sym, strat in failed_cases:
            print(f"- {sym}-{strat}")

if __name__ == "__main__":
    test_all_predictions()
