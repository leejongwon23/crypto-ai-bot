# signal_explainer.py (왕1 보완 기능: 지표 기반 진입 사유 생성)

def explain_signals(latest_row):
    """
    기술적 지표 값에 따라 텍스트 설명을 생성한다.
    - latest_row: dict 또는 Series. 'rsi', 'macd', 'boll' 키 필요
    """
    explanations = []

    # RSI 조건
    rsi = latest_row.get("rsi", 50)
    if rsi < 30:
        explanations.append("📉 RSI 과매도 구간 접근")
    elif rsi > 70:
        explanations.append("📈 RSI 과매수 상태")

    # MACD 조건
    macd = latest_row.get("macd", 0)
    if macd > 0:
        explanations.append("🔺 MACD 상승 모멘텀")
    elif macd < 0:
        explanations.append("🔻 MACD 하락 모멘텀")

    # Bollinger Band 조건 (표준화 기준값 -1 ~ 1 중심)
    boll = latest_row.get("boll", 0)
    if boll > 1:
        explanations.append("⬆️ 밴드 상단 돌파")
    elif boll < -1:
        explanations.append("⬇️ 밴드 하단 이탈")

    if not explanations:
        return "기술 지표 중립"
    return " / ".join(explanations)
