import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ 수정 코드 (통일된 설정 사용)
from config import NUM_CLASSES

# ✅ 추가
import xgboost as xgb
import numpy as np

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        weights = F.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        return context, weights


class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3, output_size=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.act = nn.GELU()
        self.fc_logits = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.norm(context)
        context = self.dropout(context)
        hidden = self.act(self.fc1(context))
        return self.fc_logits(hidden)


class CNNLSTMPricePredictor(nn.Module):
    def __init__(self, input_size, cnn_channels=64, lstm_hidden_size=128, lstm_layers=2, dropout=0.3, output_size=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden_size, lstm_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(lstm_hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_size * 2, lstm_hidden_size)
        self.act = nn.GELU()
        self.fc_logits = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.dropout(context)
        hidden = self.act(self.fc1(context))
        return self.fc_logits(hidden)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.3):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )

    def forward(self, x):
        return self.layer(x)


class TransformerPricePredictor(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.3, output_size=NUM_CLASSES):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = nn.Sequential(*[TransformerEncoderLayer(d_model, nhead, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.act = nn.GELU()
        self.fc_logits = nn.Linear(d_model // 2, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        hidden = self.act(self.fc1(x))
        return self.fc_logits(hidden)


# ✅ xgboost wrapper class
class XGBoostWrapper:
    def __init__(self, model_path):
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def predict(self, X):
        dmatrix = xgb.DMatrix(X.reshape(X.shape[0], -1))
        probs = self.model.predict(dmatrix)
        return np.argmax(probs, axis=1)


MODEL_CLASSES = {
    "lstm": LSTMPricePredictor,
    "cnn_lstm": CNNLSTMPricePredictor,
    "transformer": TransformerPricePredictor,
    "xgboost": XGBoostWrapper
}

def get_model(model_type="cnn_lstm", input_size=11, output_size=None, model_path=None):
    if model_type == "xgboost":
        if model_path is None:
            raise ValueError("XGBoost model_path must be provided.")
        return XGBoostWrapper(model_path)

    if model_type not in MODEL_CLASSES:
        print(f"[경고] 알 수 없는 모델 타입 '{model_type}', 기본 모델 cnn_lstm 사용")
    model_cls = MODEL_CLASSES.get(model_type, CNNLSTMPricePredictor)

    if output_size is None:
        from config import NUM_CLASSES
        output_size = NUM_CLASSES

    return model_cls(input_size=input_size, output_size=output_size)
