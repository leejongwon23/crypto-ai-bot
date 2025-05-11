def format_message(data):
    price = data['price']
    stop_loss = price * (1 - 0.02) if data['direction'] == '롱' else price * (1 + 0.02)

    # 전략별 최대 수익률 기준
    max_rate = {
        "단기": 50,
        "중기": 80,
        "장기": 100
    }

    rate_pct = data['rate'] * 100
    suffix = "+" if rate_pct > max_rate.get(data['strategy'], 100) else ""

    return (
        f"[{data['strategy']} 전략] {data['symbol']} {data['direction']} 추천\n"
        f"예측 수익률 구간: {rate_pct:.1f}%{suffix} "
        f"{'상승' if data['direction'] == '롱' else '하락'} 예상\n"
        f"진입가: {price:.2f} USDT\n"
        f"목표가: {data['target']:.2f} USDT (+{rate_pct:.2f}%{suffix})\n"
        f"손절가: {stop_loss:.2f} USDT (-2.00%)\n\n"
        f"신호 방향: {'상승' if data['direction'] == '롱' else '하락'}\n"
        f"신뢰도: {data['confidence']*100:.2f}%\n"
        f"추천 사유: {data['reason']}"
    )
