import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
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
        if self.training and x.size(0) == 1:
            self.eval()  # 배치 크기 1이면 에러 방지를 위해 평가모드로 변경

        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.bn(context)
        context = self.dropout(context)
        signal = torch.sigmoid(self.fc_signal(context)).squeeze(-1)
        confidence = torch.sigmoid(self.fc_confidence(context)).squeeze(-1)
        return signal, confidence

class CNNLSTMPricePredictor(nn.Module):
    def __init__(self, input_size: int, cnn_channels: int = 32, lstm_hidden_size: int = 64, lstm_layers: int = 2, dropout: float = 0.3):
        super(CNNLSTMPricePredictor, self).__init__()
        self.conv1 = nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=cnn_channels,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True)
        self.attention = Attention(lstm_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_signal = nn.Linear(lstm_hidden_size, 1)
        self.fc_confidence = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, input_size, seq_len]
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # [batch, seq_len, channels]
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.dropout(context)
        signal = torch.sigmoid(self.fc_signal(context)).squeeze(-1)
        confidence = torch.sigmoid(self.fc_confidence(context)).squeeze(-1)
        return signal, confidence

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.3):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

    def forward(self, x):
        return self.transformer(x)

class TransformerPricePredictor(nn.Module):
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer(d_model, nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc_signal = nn.Linear(d_model, 1)
        self.fc_confidence = nn.Linear(d_model, 1)

    def forward(self, x):
        if self.training and x.size(0) == 1:
            self.eval()

        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        signal = torch.sigmoid(self.fc_signal(x)).squeeze(-1)
        confidence = torch.sigmoid(self.fc_confidence(x)).squeeze(-1)
        return signal, confidence

MODEL_CLASSES = {
    "lstm": LSTMPricePredictor,
    "cnn_lstm": CNNLSTMPricePredictor,
    "transformer": TransformerPricePredictor
}

def get_model(model_type: str = "cnn_lstm", input_size: int = 11):
    model_cls = MODEL_CLASSES.get(model_type, CNNLSTMPricePredictor)
    return model_cls(input_size=input_size)

def format_prediction(signal: float, confidence: float, rate: float) -> dict:
    direction = "롱" if signal > 0.5 else "숏"
    return {
        "direction": direction,
        "confidence": float(confidence),
        "rate": float(rate)
    }
