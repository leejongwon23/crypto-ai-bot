# predict_test.py (YOPO v1.5 — 배치 예측 러너, 평가 즉시 실행, reason 비의존)
# 핵심:
#  - predict() 반환 dict의 'reason' 값에 의존하지 않음. class>=0이면 성공 처리.
#  - 예측 후 run_evaluation_once() 즉시 수행. --force-eval 플래그 유지(내부 확장 여지).
#  - 게이트 열고 닫음. 텔레그램 실패 시 조용히 스킵.

import os
import sys
import argparse
import traceback
import datetime
import pytz

from predict import predict, open_predict_gate, close_predict_gate, run_evaluation_once

# 선택적: 모델 존재 여부
try:
    from model_weight_loader import model_exists
except Exception:
    def model_exists(symbol, strategy):  # fallback
        return True

# 심볼 목록
try:
    from data.utils import SYMBOLS as _SYMBOLS
    SYMBOLS = _SYMBOLS
except Exception:
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"]

STRATEGIES = ["단기", "중기", "장기"]

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def _send_telegram(msg: str):
    try:
        from telegram_bot import send_message
        send_message(msg)
    except Exception:
        pass  # 미설정 시 무음

def run_once(strategy: str, symbols=None, force_eval: bool=False):
    print(f"\n📋 [예측 시작] 전략: {strategy} | 시각: {now_kst().strftime('%Y-%m-%d %H:%M:%S')}")
    total, ok, failed, skipped = 0, 0, 0, 0
    failed_cases = []

    target_symbols = symbols if symbols is not None else SYMBOLS

    for symbol in target_symbols:
        if not model_exists(symbol, strategy):
            skipped += 1
            print(f"⏭️ SKIP: {symbol}-{strategy} → 모델 없음")
            continue

        total += 1
        try:
            result = predict(symbol, strategy)  # dict 하나 반환
            if not isinstance(result, dict):
                failed += 1
                failed_cases.append((symbol, strategy, "반환형식 오류"))
                print(f"❌ 실패: {symbol}-{strategy} → 반환형식 오류")
                continue

            cls = int(result.get("class", result.get("predicted_class", -1)))
            exp_ret = float(result.get("expected_return", 0.0))
            model = str(result.get("model", "meta"))
            if cls < 0:
                failed += 1
                failed_cases.append((symbol, strategy, str(result.get("reason", "no_class"))))
                print(f"❌ 실패: {symbol}-{strategy} → {result.get('reason', 'no_class')}")
                continue

            print(f"✅ 완료: {symbol}-{strategy} | model={model} | class={cls} | expected≈{exp_ret:.2%}")
            ok += 1

        except Exception as e:
            failed += 1
            failed_cases.append((symbol, strategy, f"예외: {e}"))
            print(f"⚠️ 예외 발생: {symbol}-{strategy} → {e}")
            traceback.print_exc()

    print("\n📌 === 예측 요약 ===")
    print(f"▶️ 총 시도: {total}")
    print(f"✅ 완료(로그 기록됨): {ok}")
    print(f"❌ 실패: {failed}")
    print(f"⏭️ 모델 없음 SKIP: {skipped}")

    if failed_cases:
        print("\n🧨 실패 목록:")
        for sym, strat, rsn in failed_cases:
            print(f"- {sym}-{strat} → {rsn}")

    # ✅ 예측 후 즉시 평가 실행
    try:
        run_evaluation_once()
        if force_eval:
            print("⚡ 강제 평가 실행 요청 접수(내부 확장 필요 시 evaluate_predictions 확장)")
    except Exception as e:
        print(f"[⚠️ 평가 실행 실패] {e}")

    _send_telegram(f"📡 전략 {strategy} 예측 완료: 완료 {ok} / 실패 {failed} / 스킵 {skipped}")

def main():
    parser = argparse.ArgumentParser(description="Batch prediction runner (gate-aware).")
    parser.add_argument("--strategy", choices=STRATEGIES + ["all"], default="all")
    parser.add_argument("--symbols", type=str, default="", help="쉼표로 구분된 심볼 목록 (예: BTCUSDT,ETHUSDT)")
    parser.add_argument("--force-eval", action="store_true", help="평가 강제 실행 플래그")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    open_predict_gate(note="predict_test.py")
    try:
        if args.strategy == "all":
            for strat in STRATEGIES:
                run_once(strat, symbols, force_eval=args.force_eval)
        else:
            run_once(args.strategy, symbols, force_eval=args.force_eval)
    finally:
        close_predict_gate()

if __name__ == "__main__":
    main()
