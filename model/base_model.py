import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import numpy as np
from config import NUM_CLASSES
from data.utils import compute_features  # ✅ compute_features import 추가

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        weights = F.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        return context, weights

class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.4, output_size=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc_logits = nn.Linear(hidden_size//2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.norm(context)
        context = self.dropout(context)
        hidden = self.act(self.fc1(context))
        hidden = self.act(self.fc2(hidden))
        return self.fc_logits(hidden)

class CNNLSTMPricePredictor(nn.Module):
    def __init__(self, input_size, cnn_channels=128, lstm_hidden_size=256, lstm_layers=3, dropout=0.4, output_size=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden_size, lstm_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(lstm_hidden_size * 2)
        self.norm = nn.LayerNorm(lstm_hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_size * 2, lstm_hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(lstm_hidden_size, lstm_hidden_size//2)
        self.fc_logits = nn.Linear(lstm_hidden_size//2, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.norm(context)
        context = self.dropout(context)
        hidden = self.act(self.fc1(context))
        hidden = self.act(self.fc2(hidden))
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
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.4):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )

    def forward(self, x):
        return self.layer(x)

class TransformerPricePredictor(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dropout=0.4, output_size=NUM_CLASSES):
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

class XGBoostWrapper:
    def __init__(self, model_path):
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def predict(self, X):
        dmatrix = xgb.DMatrix(X.reshape(X.shape[0], -1))
        probs = self.model.predict(dmatrix)
        return np.argmax(probs, axis=1)


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        x = x.squeeze(1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.unsqueeze(1)
        return decoded


MODEL_CLASSES = {
    "lstm": LSTMPricePredictor,
    "cnn_lstm": CNNLSTMPricePredictor,
    "cnn": CNNLSTMPricePredictor,
    "transformer": TransformerPricePredictor,
    "xgboost": XGBoostWrapper,
    "autoencoder": AutoEncoder
}

def get_model(model_type="cnn_lstm", input_size=None, output_size=None, model_path=None, features=None):
    from data.utils import compute_features, get_kline_by_strategy  # ✅ df 확보 위해 추가 import

    if model_type == "xgboost":
        if model_path is None:
            raise ValueError("XGBoost model_path must be provided.")
        return XGBoostWrapper(model_path)

    if model_type not in MODEL_CLASSES:
        print(f"[경고] 알 수 없는 모델 타입 '{model_type}', 기본 모델 cnn_lstm 사용")
        model_cls = CNNLSTMPricePredictor
    else:
        model_cls = MODEL_CLASSES[model_type]

    if output_size is None:
        from config import NUM_CLASSES
        output_size = NUM_CLASSES

    # ✅ features가 제공되면 input_size를 features.shape[2]로 자동 지정
    if input_size is None:
        if features is not None:
            input_size = features.shape[2]
            print(f"[info] input_size 자동설정(features): {input_size}")
        else:
            # ✅ compute_features() 호출 시 df 포함하도록 변경
            try:
                sample_df_df = get_kline_by_strategy("BTCUSDT", "단기")
                if sample_df_df is not None and not sample_df_df.empty:
                    sample_df = compute_features("BTCUSDT", sample_df_df, "단기")
                    input_size = sample_df.drop(columns=["timestamp", "strategy"], errors="ignore").shape[1]
                    print(f"[info] input_size auto-calculated from compute_features: {input_size}")
                else:
                    input_size = 11  # fallback 기본값
                    print(f"[⚠️ input_size fallback=11] get_kline_by_strategy 반환 None 또는 empty")
            except Exception as e:
                input_size = 11  # fallback 기본값
                print(f"[⚠️ input_size fallback=11] compute_features 예외 발생: {e}")

    try:
        model = model_cls(input_size=input_size, output_size=output_size)
    except Exception as e:
        print(f"[⚠️ get_model 예외] {e}")
        print(f"[Fallback] input_size=14로 재시도")
        try:
            model = model_cls(input_size=14, output_size=output_size)
        except Exception as e2:
            print(f"[❌ get_model 실패] {e2}")
            raise e2

    return model




