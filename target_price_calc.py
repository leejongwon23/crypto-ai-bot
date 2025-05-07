# target_price_calc.py (왕1 보완 기능: 목표가/손절가 자동 계산)

def calculate_targets(entry_price: float, volatility: float = 0.02):
    """
    목표가와 손절가를 계산한다.
    - entry_price: 현재 진입가
    - volatility: 변동성 기준 (기본값 2%)
    """
    take_profit = entry_price * (1 + volatility * 1.5)  # 예: 3% 상승 목표
    stop_loss = entry_price * (1 - volatility)          # 예: 2% 손실 제한

    return round(take_profit, 2), round(stop_loss, 2)


# 사용 예시 (예: 진입가 1000, 변동성 2%)
if __name__ == "__main__":
    tp, sl = calculate_targets(1000)
    print("목표가:", tp, "손절가:", sl)
