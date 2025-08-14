import os
import csv
import json
import traceback
import datetime
import pytz
import math
import pandas as pd

from predict import predict
from data.utils import SYMBOLS, get_kline_by_strategy
from logger import log_prediction
from telegram_bot import send_message

# 현재 KST 시각
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

# 전략별 변동성 기준
STRATEGY_VOL = {"단기": 0.003, "중기": 0.005, "장기": 0.008}

# 로그 경로
AUDIT_LOG = "/persistent/logs/prediction_audit.csv"
FAILURE_LOG = "/persistent/logs/failure_count.csv"
PREDICTION_LOG = "/persistent/logs/prediction_log.csv"
os.makedirs("/persistent/logs", exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 성공률 필터 (성공률 ≥65% + 최소 10회 기록)
# ──────────────────────────────────────────────────────────────
def check_prediction_filter(strategy, min_success_rate=0.65, min_samples=10):
    try:
        if not os.path.exists(PREDICTION_LOG):
            return False
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
# 텔레그램 메시지 포맷
# ──────────────────────────────────────────────────────────────
def format_message(data):
    def safe_float(value, default=0.0):
        try:
            if value is None or (isinstance(value, str) and not str(value).strip()):
                return default
            val = float(value)
            return val if not math.isnan(val) else default
        except:
            return default

    price = safe_float(data.get("price"), 0.0)
    direction = data.get("direction", "롱")
    strategy = data.get("strategy", "전략")
    symbol = data.get("symbol", "종목")
    success_rate = safe_float(data.get("success_rate"), 0.0)
    rate = safe_float(data.get("rate"), 0.0)
    reason = str(data.get("reason", "-")).strip()
    score = data.get("score", None)
    volatility = str(data.get("volatility", "False")).lower() in ["1", "true", "yes"]

    target = price * (1 + rate) if direction == "롱" else price * (1 - rate)
    stop_loss = price * (1 - 0.02) if direction == "롱" else price * (1 + 0.02)

    rate_pct = abs(rate) * 100
    success_rate_pct = success_rate * 100
    dir_str = "상승" if direction == "롱" else "하락"
    vol_tag = "⚡ " if volatility else ""

    message = (
        f"{vol_tag}{'📈' if direction == '롱' else '📉'} "
        f"[{strategy} 전략] {symbol} {direction} 추천\n"
        f"🎯 예상 수익률: {rate_pct:.2f}% ({dir_str} 예상)\n"
        f"💰 진입가: {price:.4f} USDT\n"
        f"🎯 목표가: {target:.4f} USDT\n"
        f"🛡 손절가: {stop_loss:.4f} USDT (-2.00%)\n\n"
        f"📊 최근 전략 성공률: {success_rate_pct:.2f}%"
    )

    if isinstance(score, (float, int)) and not math.isnan(score):
        message += f"\n🏆 스코어: {score:.5f}"

    message += f"\n💡 추천 사유: {reason}\n\n🕒 (기준시각: {now_kst().strftime('%Y-%m-%d %H:%M:%S')} KST)"
    return message

# ──────────────────────────────────────────────────────────────
# 감사 로그 기록
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
# 변동성 높은 심볼 추출
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
            if r_std >= th and (r_std / (b_std + 1e-8)) >= 1.2:
                result.append({"symbol": symbol, "volatility": r_std})
        except Exception as e:
            print(f"[ERROR] 변동성 계산 실패: {symbol}-{strategy}: {e}")
    return sorted(result, key=lambda x: -x["volatility"])

# ──────────────────────────────────────────────────────────────
# 예측 실행 루프
# ──────────────────────────────────────────────────────────────
def run_prediction_loop(strategy, symbols, source="일반", allow_prediction=True):
    print(f"[예측 시작 - {strategy}] {len(symbols)}개 심볼")
    results, fmap = [], load_failure_count()

    for item in symbols:
        symbol = item["symbol"]

        if not allow_prediction:
            log_audit(symbol, strategy, "예측 생략", f"예측 차단됨 (source={source})")
            continue

        try:
            model_dir = "/persistent/models"
            model_count = len([
                f for f in os.listdir(model_dir)
                if f.startswith(f"{symbol}_{strategy}_") and (f.endswith(".pt") or f.endswith(".meta.json"))
            ])
            if model_count == 0:
                log_audit(symbol, strategy, None, "모델 없음")
                continue

            pred_results = predict(symbol, strategy, source=source)
            if not isinstance(pred_results, list):
                pred_results = [pred_results]

            if not pred_results:
                log_audit(symbol, strategy, None, "predict() 결과 없음")
                continue

            for result in pred_results:
                if not isinstance(result, dict):
                    log_audit(symbol, strategy, str(result), "예측 반환 형식 오류")
                    continue

                log_prediction(
                    symbol=result.get("symbol", symbol),
                    strategy=result.get("strategy", strategy),
                    direction=result.get("direction", "class"),
                    entry_price=float(result.get("price", 0.0)),
                    target_price=float(result.get("price", 0.0)) * (1 + float(result.get("expected_return", 0.0))),
                    timestamp=result.get("timestamp", now_kst().isoformat()),
                    model=result.get("model", "unknown"),
                    success=True,
                    reason=result.get("reason", "예측 기록"),
                    rate=float(result.get("expected_return", 0.0)),
                    return_value=float(result.get("expected_return", 0.0)),
                    volatility=item.get("volatility", 0) > 0,
                    source=source,
                    predicted_class=int(result.get("class", -1)),
                    label=int(result.get("class", -1)),
                )
                results.append(result)
                fmap[f"{symbol}-{strategy}"] = 0

        except Exception as e:
            print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
            traceback.print_exc()

    save_failure_count(fmap)
    return results

# ──────────────────────────────────────────────────────────────
# 메인 엔트리 — 배치 예측
# ──────────────────────────────────────────────────────────────
def main(strategy, symbols=None, force=False, allow_prediction=True):
    print(f"\n📋 [예측 시작] 전략: {strategy} | 시각: {now_kst().strftime('%Y-%m-%d %H:%M:%S')}")
    target_symbols = symbols if symbols is not None else get_symbols_by_volatility(strategy)
    if not target_symbols:
        print(f"[INFO] {strategy} 대상 심볼이 없습니다")
        return

    results = run_prediction_loop(strategy, target_symbols, source="배치", allow_prediction=allow_prediction)

    if check_prediction_filter(strategy):
        for r in results:
            try:
                send_message(format_message(r))
            except Exception as e:
                print(f"[텔레그램 전송 실패] {e}")
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
