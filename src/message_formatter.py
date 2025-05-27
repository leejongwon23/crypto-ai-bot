import datetime
import pytz
import math

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")

def format_message(data):
    def safe_float(value, default=0.0):
        try:
            if value is None or (isinstance(value, str) and not value.strip()):
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
    target = safe_float(data.get("target"), 0.0)
    reason = str(data.get("reason", "-")).strip()
    score = data.get("score", None)
    reversed_signal = data.get("reversed", False)

    stop_loss = price * (1 - 0.02) if direction == "롱" else price * (1 + 0.02)
    rate_pct = abs(rate) * 100
    success_rate_pct = success_rate * 100
    dir_str = "상승" if direction == "롱" else "하락"
    title = "[반전 전략] " if reversed_signal else ""

    message = (
        f"{title}[{strategy} 전략] {symbol} {direction} 추천\n"
        f"예상 수익률: {rate_pct:.2f}% {dir_str} 예상\n"
        f"진입가: {price:.4f} USDT\n"
        f"목표가: {target:.4f} USDT\n"
        f"손절가: {stop_loss:.4f} USDT (-2.00%)\n\n"
        f"신호 방향: {dir_str}\n"
        f"성공률: {success_rate_pct:.2f}%"
    )

    if isinstance(score, (float, int)) and not math.isnan(score):
        message += f"\n스코어: {score:.5f}"

    message += f"\n추천 사유: {reason}\n\n(기준시각: {now_kst()} KST)"
    return message
