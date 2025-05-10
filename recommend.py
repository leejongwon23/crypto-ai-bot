import datetime
from telegram_bot import send_message
from train import predict, STRATEGY_GAIN_LEVELS
from logger import log_prediction, evaluate_predictions
from data.utils import SYMBOLS, get_realtime_prices

# 손절가는 1.4에서 직접 계산
def format_message(data):
    price = data['price']
    stop_loss = price * (1 - 0.02) if data['direction'] == '롱' else price * (1 + 0.02)

    return (
        f"[{data['strategy']} 전략] {data['symbol']} {data['direction']} 추천\n"
        f"예측 수익률 구간: {data['rate']*100:.1f}% "
        f"{'상승' if data['direction'] == '롱' else '하락'} 예상\n"
        f"진입가: {price:.2f} USDT\n"
        f"목표가: {data['target']:.2f} USDT (+{data['rate']*100:.2f}%)\n"
        f"손절가: {stop_loss:.2f} USDT (-2.00%)\n\n"
        f"신호 방향: {'상승' if data['direction'] == '롱' else '하락'}\n"
        f"신뢰도: {data['confidence']*100:.2f}%\n"
        f"추천 사유: {data['reason']}"
    )

def get_price_now(symbol):
    prices = get_realtime_prices()
    return prices.get(symbol)

def main():
    evaluate_predictions(get_price_now)
    for strategy in STRATEGY_GAIN_LEVELS:
        for symbol in SYMBOLS:
            try:
                result = predict(symbol, strategy)
                if result:
                    log_prediction(
                        symbol=result["symbol"],
                        strategy=result["strategy"],
                        direction=result["direction"],
                        entry_price=result["price"],
                        target_price=result["target"],
                        timestamp=datetime.datetime.utcnow().isoformat(),
                        confidence=result["confidence"]
                    )
                    if result["confidence"] > 0.7:
                        msg = format_message(result)
                        send_message(msg)
            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")
