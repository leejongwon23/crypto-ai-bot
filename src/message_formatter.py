import datetime
import pytz

def now_kst():
    return datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")

def format_message(data):
    price = data['price']
    stop_loss = price * (1 - 0.02) if data['direction'] == '롱' else price * (1 + 0.02)

    rate_pct = data['rate'] * 100
    direction = '상승' if data['direction'] == '롱' else '하락'
    success_rate_pct = data.get("success_rate", 0.5) * 100

    return (
        f"[{data['strategy']} 전략] {data['symbol']} {data['direction']} 추천\n"
        f"예상 수익률: {rate_pct:.2f}% {direction} 예상\n"
        f"진입가: {price:.2f} USDT\n"
        f"목표가: {data['target']:.2f} USDT (+{rate_pct:.2f}%)\n"
        f"손절가: {stop_loss:.2f} USDT (-2.00%)\n\n"
        f"신호 방향: {direction}\n"
        f"신뢰도: {data['confidence']*100:.2f}%\n"
        f"성공률: {success_rate_pct:.2f}%\n"
        f"추천 사유: {data['reason']}\n\n"
        f"(기준시각: {now_kst()} KST)"
    )
