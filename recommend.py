# === recommend.py (수정본) ===
import os
import csv
import time
import json
import traceback
import datetime
import pytz

from predict import predict
from data.utils import SYMBOLS, get_kline_by_strategy
from logger import log_prediction
from telegram_bot import send_message

now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

STRATEGY_VOL = {"단기": 0.003, "중기": 0.005, "장기": 0.008}
AUDIT_LOG = "/persistent/logs/prediction_audit.csv"
FAILURE_LOG = "/persistent/logs/failure_count.csv"
MESSAGE_LOG = "/persistent/logs/message_log.csv"
PREDICTION_LOG = "/persistent/logs/prediction_log.csv"  # ✅ 예측 기록 로그 경로
os.makedirs("/persistent/logs", exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 성공률 필터 (성공률 65% 이상 + 최소 10회 기록 시 True)
# ──────────────────────────────────────────────────────────────
def check_prediction_filter(strategy, min_success_rate=0.65, min_samples=10):
    try:
        if not os.path.exists(PREDICTION_LOG):
            return False
        import pandas as pd
        df = pd.read_csv(PREDICTION_LOG)
        df = df[df["strategy"] == strategy]
        if len(df) < min_samples:
            return False
        success_rate = df["success"].mean()
        return success_rate >= min_success_rate
    except Exception as e:
        print(f"[prediction_filter 예외] {e}")
        return False

# ──────────────────────────────────────────────────────────────
# 감사 로그
# ──────────────────────────────────────────────────────────────
def log_audit(symbol, strategy, result, status):
    try:
        with open(AUDIT_LOG, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "symbol", "strategy", "result", "status"])
            if f.tell() == 0:
                w.writeheader()
            w.writerow({
                "timestamp": now_kst().isoformat(),
                "symbol": symbol or "UNKNOWN",
                "strategy": strategy or "알수없음",
                "result": str(result),
                "status": status
            })
    except Exception as e:
        print(f"[log_audit 오류] {e}")

# ──────────────────────────────────────────────────────────────
# 실패 카운트 로드/저장
# ──────────────────────────────────────────────────────────────
def load_failure_count():
    if not os.path.exists(FAILURE_LOG):
        return {}
    with open(FAILURE_LOG, "r", encoding="utf-8-sig") as f:
        return {f"{r['symbol']}-{r['strategy']}": int(r["failures"]) for r in csv.DictReader(f)}

def save_failure_count(fmap):
    with open(FAILURE_LOG, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "strategy", "failures"])
        w.writeheader()
        for k, v in fmap.items():
            s, strat = k.split("-")
            w.writerow({"symbol": s, "strategy": strat, "failures": v})

# ──────────────────────────────────────────────────────────────
# 전략별 변동성 높은 심볼 선별
# ──────────────────────────────────────────────────────────────
def get_symbols_by_volatility(strategy):
    th = STRATEGY_VOL.get(strategy, 0.003)
    result = []
    for symbol in SYMBOLS:
        try:
            df = get_kline_by_strategy(symbol, strategy)
            if df is None or len(df) < 60:
                continue
            r_std = df["close"].pct_change().rolling(20).std().iloc[-1]
            b_std = df["close"].pct_change().rolling(60).std().iloc[-1]
            is_volatile = r_std >= th
            is_rising = (r_std / (b_std + 1e-8)) >= 1.2
            if is_volatile and is_rising:
                result.append({"symbol": symbol, "volatility": r_std})
        except Exception as e:
            print(f"[ERROR] 변동성 계산 실패: {symbol}-{strategy}: {e}")
    return sorted(result, key=lambda x: -x["volatility"])

# ──────────────────────────────────────────────────────────────
# 예측 실행 루프 (심볼 리스트)
# ──────────────────────────────────────────────────────────────
def run_prediction_loop(strategy, symbols, source="일반", allow_prediction=True):
    print(f"[예측 시작 - {strategy}] {len(symbols)}개 심볼")
    results, fmap = [], load_failure_count()

    for item in symbols:
        symbol = item["symbol"]
        vol = item.get("volatility", 0)

        if not allow_prediction:
            log_audit(symbol, strategy, "예측 생략", f"예측 차단됨 (source={source})")
            continue

        try:
            # 모델 존재 여부 대략 체크
            model_dir = "/persistent/models"
            model_count = len([
                f for f in os.listdir(model_dir)
                if f.startswith(f"{symbol}_{strategy}_") and (f.endswith(".pt") or f.endswith(".meta.json"))
            ])
            if model_count == 0:
                log_audit(symbol, strategy, None, "모델 없음")
                log_prediction(
                    symbol=symbol,
                    strategy=strategy,
                    direction="예측실패",
                    entry_price=0,
                    target_price=0,
                    timestamp=now_kst().isoformat(),
                    model="unknown",
                    success=False,
                    reason="모델 없음",
                    rate=0.0,
                    return_value=0.0,
                    volatility=False,
                    source=source,
                    predicted_class=-1,
                    label=-1,
                )
                continue

            # 실제 예측 실행
            pred_results = predict(symbol, strategy, source=source)
            if not isinstance(pred_results, list):
                pred_results = [pred_results]

            # 결과 처리
            if not pred_results:
                log_audit(symbol, strategy, None, "predict() 결과 없음")
                log_prediction(
                    symbol=symbol,
                    strategy=strategy,
                    direction="예측실패",
                    entry_price=0,
                    target_price=0,
                    timestamp=now_kst().isoformat(),
                    model="unknown",
                    success=False,
                    reason="predict() 결과 없음",
                    rate=0.0,
                    return_value=0.0,
                    volatility=False,
                    source=source,
                    predicted_class=-1,
                    label=-1,
                )
                continue

            for result in pred_results:
                if not isinstance(result, dict):
                    log_audit(symbol, strategy, str(result), "예측 반환 형식 오류")
                    continue

                # 실패/에러 케이스
                if result.get("reason") in ["모델 없음", "데이터 부족", "feature 부족"]:
                    reason = result.get("reason", "예측 실패")
                    pred_class_val = -1
                    log_prediction(
                        symbol=symbol,
                        strategy=strategy,
                        direction="예측실패",
                        entry_price=0,
                        target_price=0,
                        timestamp=now_kst().isoformat(),
                        model=result.get("model", "unknown"),
                        success=False,
                        reason=reason,
                        rate=0.0,
                        return_value=0.0,
                        volatility=False,
                        source=source,
                        predicted_class=pred_class_val,
                        label=pred_class_val,
                    )
                    log_audit(symbol, strategy, result, reason)
                    continue

                # 정상 케이스
                pred_class_val = int(result.get("class", -1))
                expected_ret = float(result.get("expected_return", 0.0))
                entry_price = float(result.get("price", 0.0))
                ts = result.get("timestamp", now_kst().isoformat())
                model_name = result.get("model", "unknown")
                src = result.get("source", source)

                log_prediction(
                    symbol=result.get("symbol", symbol),
                    strategy=result.get("strategy", strategy),
                    direction=f"class-{pred_class_val}",
                    entry_price=entry_price,
                    target_price=entry_price * (1 + expected_ret) if entry_price > 0 else 0,
                    timestamp=ts,
                    model=model_name,
                    success=True,  # 최종 평가는 evaluate가 결정
                    reason=result.get("reason", "예측 기록"),
                    rate=expected_ret,
                    return_value=expected_ret,
                    volatility=vol > 0,
                    source=src,
                    predicted_class=pred_class_val,
                    label=pred_class_val,
                )
                log_audit(symbol, strategy, result, "예측 기록 완료")

                results.append(result)
                fmap[f"{symbol}-{strategy}"] = 0  # 실패 카운터 리셋

        except Exception as e:
            print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
            traceback.print_exc()
            log_audit(symbol, strategy, None, f"예측 예외: {e}")

    save_failure_count(fmap)
    return results

# ──────────────────────────────────────────────────────────────
# 단일 심볼 즉시 실행
# ──────────────────────────────────────────────────────────────
def run_prediction(symbol, strategy, source="단일"):
    print(f">>> [run_prediction] {symbol} - {strategy} 예측 시작")
    model_dir = "/persistent/models"

    for mt in ["transformer", "cnn_lstm", "lstm"]:
        pt_file = f"{symbol}_{strategy}_{mt}.pt"
        meta_file = f"{symbol}_{strategy}_{mt}.meta.json"
        if os.path.exists(os.path.join(model_dir, pt_file)) and os.path.exists(os.path.join(model_dir, meta_file)):
            run_prediction_loop(strategy, [{"symbol": symbol, "model_type": mt}], source=source, allow_prediction=True)
            return

    print(f"[run_prediction] {symbol}-{strategy} 가능한 모델 없음")
    log_prediction(
        symbol=symbol,
        strategy=strategy,
        direction="예측실패",
        entry_price=0,
        target_price=0,
        timestamp=now_kst().isoformat(),
        model="unknown",
        success=False,
        reason="모델 없음",
        rate=0.0,
        return_value=0.0,
        volatility=False,
        source=source,
        predicted_class=-1,
        label=-1,
    )

# ──────────────────────────────────────────────────────────────
# 메인 엔트리 — 배치 예측
# ──────────────────────────────────────────────────────────────
def main(strategy, symbols=None, force=False, allow_prediction=True):
    print(f"\n📋 [예측 시작] 전략: {strategy} | 시각: {now_kst().strftime('%Y-%m-%d %H:%M:%S')}")
    target_symbols = symbols if symbols is not None else get_symbols_by_volatility(strategy)

    if not target_symbols:
        print(f"[INFO] {strategy} 대상 심볼이 비었습니다")
        return

    results = run_prediction_loop(strategy, target_symbols, source="배치", allow_prediction=allow_prediction)
    succ = sum(1 for r in results if isinstance(r, dict))
    fail = len(target_symbols) - succ
    print(f"[요약] {strategy} 실행 결과: 성공기록 {succ} / 실패·스킵 {fail}")

    # ✅ 텔레그램 전송 필터 적용
    if check_prediction_filter(strategy):
        try:
            send_message(f"📡 전략 {strategy} 예측 완료: 기록 {succ} / 스킵 {fail}")
        except Exception:
            pass
    else:
        print(f"[알림 생략] {strategy} — 성공률 65% 이상 + 최소 10회 조건 미충족")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="단기", choices=["단기", "중기", "장기"])
    parser.add_argument("--allow_prediction", action="store_true", default=True)
    args = parser.parse_args()

    try:
        main(args.strategy, allow_prediction=args.allow_prediction)
    except Exception as e:
        print(f"[❌ 예측 실패] {e}")
        traceback.print_exc()
