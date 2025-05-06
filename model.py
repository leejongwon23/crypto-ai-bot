def analyze_coin(symbol, candles, backtest=False):
    # 기존 로직은 유지
    ...

    # 백테스트용 현재가 설정
    current_price = candles[-1]['close'] if backtest else candles[-1]['close']  # 동일하나 구조유지용

    # 기존 텍스트 출력 부분에서 진입가 대신 current_price 사용
    message = f"""
📌 코인: {symbol}
📈 진입가: {round(current_price, 3)} USDT
🎯 목표가: {round(target_price, 3)} USDT
🛑 손절가: {round(stop_loss, 3)} USDT
📊 전략: {strategy_type} / {expected_return}%
📅 정확도 사유: {reason}
"""
    return message.strip()
