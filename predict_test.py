# test_all_predictions.py
import os
import sys
import argparse
import traceback
import datetime
import pytz

from predict import predict, open_predict_gate, close_predict_gate

# 선택적: 모델 존재 여부 (없으면 항상 True로 간주해 실행)
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
        # 텔레그램 미설정 시 조용히 스킵
        pass

def run_once(strategy: str, symbols=None):
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

            # predict() 정상 완료 시 result.reason == "predicted" (성공/실패 평가는 별도 루프에서)
            reason = str(result.get("reason", ""))
            if reason and reason != "predicted":
                failed += 1
                failed_cases.append((symbol, strategy, reason))
                print(f"❌ 실패: {symbol}-{strategy} → {reason}")
                continue

            cls = result.get("class", result.get("predicted_class", -1))
            exp_ret = result.get("expected_return", 0.0)
            model = result.get("model", "meta")
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

    _send_telegram(f"📡 전략 {strategy} 예측 완료: 완료 {ok} / 실패 {failed} / 스킵 {skipped}")

def main():
    parser = argparse.ArgumentParser(description="Batch prediction runner (gate-aware).")
    parser.add_argument("--strategy", choices=STRATEGIES + ["all"], default="all")
    parser.add_argument("--symbols", type=str, default="", help="쉼표로 구분된 심볼 목록 (예: BTCUSDT,ETHUSDT)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    # 예측 게이트 열기(학습 블록 외부에서만 예측 허용)
    open_predict_gate(note="test_all_predictions.py")
    try:
        if args.strategy == "all":
            for strat in STRATEGIES:
                run_once(strat, symbols)
        else:
            run_once(args.strategy, symbols)
    finally:
        close_predict_gate()

if __name__ == "__main__":
    main()
