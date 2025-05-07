from telegram_bot import send_message  # 텔레그램 전송 기능 추가

import torch
import torch.nn.functional as F
import numpy as np
from model.base_model import LSTMPricePredictor
from data.utils import SYMBOLS, STRATEGY_CONFIG, get_kline_by_strategy, compute_features, get_realtime_prices, get_long_short_ratio, get_trade_strength

STRATEGY_GAIN_LEVELS = {
    "단기": [0.05, 0.07, 0.10],
    "중기": [0.10, 0.20, 0.30],
    "장기": [0.15, 0.30, 0.60]
}
STOP_LOSS_PCT = 0.03
WINDOW = 30
DEVICE = torch.device("cpu")

def get_model(symbol, strategy, input_size, num_classes):
    class DualGainClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size, 128, 3, batch_first=True, dropout=0.3)
            self.attn = torch.nn.Linear(128, 1)
            self.bn = torch.nn.BatchNorm1d(128)
            self.drop = torch.nn.Dropout(0.3)
            self.fc = torch.nn.Linear(128, num_classes)
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            w = F.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
            context = torch.sum(lstm_out * w.unsqueeze(-1), dim=1)
            context = self.bn(context)
            context = self.drop(context)
            return self.fc(context)

    num_classes = len(STRATEGY_GAIN_LEVELS[strategy]) * 2
    model = DualGainClassifier()
    path = f"models/{symbol}_{strategy}_dual.pt"
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict(symbol, strategy):
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < WINDOW + 1:
        return None

    features = compute_features(df)
    if len(features) < WINDOW + 1:
        return None

    X = features.iloc[-WINDOW:].values
    X = np.expand_dims(X, axis=0)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    model = get_model(symbol, strategy, input_size=X.shape[2], num_classes=len(STRATEGY_GAIN_LEVELS[strategy]) * 2)
    output = model(X_tensor)
    probs = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
    top_idx = np.argmax(probs)
    confidence = probs[top_idx]

    levels = STRATEGY_GAIN_LEVELS[strategy]
    class_count = len(levels)
    if top_idx < class_count:
        direction = "롱"
        rate = levels[top_idx]
    else:
        direction = "숏"
        rate = levels[top_idx - class_count]

    price = df["close"].iloc[-1]
    if direction == "롱":
        target = price * (1 + rate)
        stop = price * (1 - STOP_LOSS_PCT)
    else:
        target = price * (1 - rate)
        stop = price * (1 + STOP_LOSS_PCT)

    rsi = features["rsi"].iloc[-1]
    macd = features["macd"].iloc[-1]
    boll = features["bollinger"].iloc[-1]
    reason = []
    if rsi < 30: reason.append("RSI 과매도")
    elif rsi > 70: reason.append("RSI 과매수")
    if macd > 0: reason.append("MACD 상승 전환")
    else: reason.append("MACD 하락 전환")
    if boll > 1: reason.append("볼린저 상단 돌파")
    elif boll < -1: reason.append("볼린저 하단 이탈")

    return {
        "symbol": symbol,
        "strategy": strategy,
        "direction": direction,
        "price": price,
        "target": target,
        "stop": stop,
        "confidence": confidence,
        "rate": rate,
        "reason": ", ".join(reason)
    }

def format_message(data):
    return (
        f"[{data['strategy']} 전략] {data['symbol']} {data['direction']} 추천\n"
        f"예측 수익률 구간: {data['rate']*100:.1f}% "
        f"{'상승' if data['direction'] == '롱' else '하락'} 예상\n"
        f"진입가: {data['price']:.2f} USDT\n"
        f"목표가: {data['target']:.2f} USDT (+{data['rate']*100:.2f}%)\n"
        f"손절가: {data['stop']:.2f} USDT (-3.00%)\n\n"
        f"신호 방향: {'상승' if data['direction'] == '롱' else '하락'}\n"
        f"신뢰도: {data['confidence']*100:.2f}%\n"
        f"추천 사유: {data['reason']}"
    )

def main():
    for strategy in STRATEGY_GAIN_LEVELS:
        for symbol in SYMBOLS:
            try:
                result = predict(symbol, strategy)
                if result:
                    message = format_message(result)
                    print(message)
                    send_message(message)
                    print("-" * 80)
            except Exception as e:
                print(f"[ERROR] {symbol}-{strategy} 예측 실패: {e}")

if __name__ == "__main__":
    main()
