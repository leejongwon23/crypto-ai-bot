import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 16  # ✅ 클래스 개수 고정

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        weights = self.attn(lstm_out).squeeze(-1)
        weights = F.softmax(weights, dim=1)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        return context, weights


class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.3):
        super(LSTMPricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.attention = Attention(hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.act = nn.GELU()
        self.fc_logits = nn.Linear(hidden_size // 2, NUM_CLASSES)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.bn(context)
        context = self.dropout(context)
        hidden = self.act(self.fc1(context))
        logits = self.fc_logits(hidden)
        return logits


class CNNLSTMPricePredictor(nn.Module):
    def __init__(self, input_size: int, cnn_channels: int = 32, lstm_hidden_size: int = 64, lstm_layers: int = 2, dropout: float = 0.3):
        super(CNNLSTMPricePredictor, self).__init__()
        self.conv1 = nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden_size, lstm_layers, batch_first=True)
        self.attention = Attention(lstm_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_size, lstm_hidden_size // 2)
        self.act = nn.GELU()
        self.fc_logits = nn.Linear(lstm_hidden_size // 2, NUM_CLASSES)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.dropout(context)
        hidden = self.act(self.fc1(context))
        logits = self.fc_logits(hidden)
        return logits


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.3):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

    def forward(self, x):
        return self.layer(x)


class TransformerPricePredictor(nn.Module):
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.encoder = nn.Sequential(*[TransformerEncoderLayer(d_model, nhead, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.act = nn.GELU()
        self.fc_logits = nn.Linear(d_model // 2, NUM_CLASSES)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        hidden = self.act(self.fc1(x))
        logits = self.fc_logits(hidden)
        return logits


MODEL_CLASSES = {
    "lstm": LSTMPricePredictor,
    "cnn_lstm": CNNLSTMPricePredictor,
    "transformer": TransformerPricePredictor
}


def get_model(model_type: str = "cnn_lstm", input_size: int = 11):
    if model_type not in MODEL_CLASSES:
        print(f"[경고] 알 수 없는 모델 타입 '{model_type}', 기본 모델 cnn_lstm 사용")
    model_cls = MODEL_CLASSES.get(model_type, CNNLSTMPricePredictor)
    return model_cls(input_size=input_size)
