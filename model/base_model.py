import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: [batch_size, seq_len, hidden_size]
        weights = self.attn(lstm_out).squeeze(-1)  # [batch_size, seq_len]
        weights = F.softmax(weights, dim=1)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_size]
        return context, weights

class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.3):
        super(LSTMPricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.attention = Attention(hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_signal = nn.Linear(hidden_size, 1)
        self.fc_confidence = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden]
        context, _ = self.attention(lstm_out)  # [batch, hidden]
        context = self.bn(context)
        context = self.dropout(context)
        signal = torch.sigmoid(self.fc_signal(context)).squeeze(-1)       # 상승/하락 확률
        confidence = torch.sigmoid(self.fc_confidence(context)).squeeze(-1)  # 예측의 신뢰도
        return signal, confidence
