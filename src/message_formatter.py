def format_message(data):
    return (
        f"[{data['strategy']} 전략] {data['symbol']} {data['direction']} 추천\n"
        f"예측 수익률 구간: {data['rate']*100:.1f}% "
        f"{'상승' if data['direction'] == '롱' else '하락'} 예상\n"
        f"진입가: {data['price']:.2f} USDT\n"
        f"목표가: {data['target']:.2f} USDT (+{data['rate']*100:.2f}%)\n"
        f"손절가: {data['stop']:.2f} USDT (-2.00%)"
    )
