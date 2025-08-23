import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import numpy as np
from config import get_NUM_CLASSES, get_FEATURE_INPUT_SIZE

NUM_CLASSES = get_NUM_CLASSES()
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout,
                            batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_logits = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x, params=None):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.norm(context)
        context = self.dropout(context)

        if params is None:
            hidden = self.act(self.fc1(context))
            hidden = self.act(self.fc2(hidden))
            logits = self.fc_logits(hidden)
        else:
            hidden = F.gelu(F.linear(context, params['fc1.weight'], params['fc1.bias']))
            hidden = F.gelu(F.linear(hidden, params['fc2.weight'], params['fc2.bias']))
            logits = F.linear(hidden, params['fc_logits.weight'], params['fc_logits.bias'])
        return logits

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
        self.fc2 = nn.Linear(lstm_hidden_size, lstm_hidden_size // 2)
        self.fc_logits = nn.Linear(lstm_hidden_size // 2, output_size)

    def forward(self, x, params=None):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.norm(context)
        context = self.dropout(context)

        if params is None:
            hidden = self.act(self.fc1(context))
            hidden = self.act(self.fc2(hidden))
            logits = self.fc_logits(hidden)
        else:
            hidden = F.gelu(F.linear(context, params['fc1.weight'], params['fc1.bias']))
            hidden = F.gelu(F.linear(hidden, params['fc2.weight'], params['fc2.bias']))
            logits = F.linear(hidden, params['fc_logits.weight'], params['fc_logits.bias'])
        return logits

class TransformerPricePredictor(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dropout=0.4, output_size=None, mode="classification"):
        super().__init__()
        self.mode = mode
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=dropout, activation='gelu', batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if self.mode == "classification":
            self.fc1 = nn.Linear(d_model, d_model // 2)
            self.act = nn.GELU()
            self.fc_logits = nn.Linear(d_model // 2, output_size)
        elif self.mode == "reconstruction":
            self.decoder = nn.Linear(d_model, output_size)

    def forward(self, x, params=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm(x)
        x = self.dropout(x)

        if self.mode == "classification":
            x = x.mean(dim=1)
            if params is None:
                x = self.act(self.fc1(x))
                logits = self.fc_logits(x)
            else:
                x = F.gelu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
                logits = F.linear(x, params['fc_logits.weight'], params['fc_logits.bias'])
            return logits
        elif self.mode == "reconstruction":
            return self.decoder(x)

# PositionalEncoding, XGBoostWrapper, AutoEncoder, get_model 함수는 기존과 동일

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

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
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
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
    from config import FEATURE_INPUT_SIZE, NUM_CLASSES

    # ✅ XGBoost 비활성화 → 자동 대체
    if model_type == "xgboost":
        print("[⚠️ get_model] 현재 XGBoost 모델은 비활성화 상태입니다. cnn_lstm 대체 사용")
        model_type = "cnn_lstm"  # 자동 대체

    # ✅ MODEL_CLASSES 직접 사용 (외부 import 제거)
    if model_type not in MODEL_CLASSES:
        print(f"[경고] 알 수 없는 모델 타입 '{model_type}', 기본 모델 cnn_lstm 사용")
        model_cls = CNNLSTMPricePredictor
    else:
        model_cls = MODEL_CLASSES[model_type]

    # ✅ 출력 클래스 수 설정
    if output_size is None:
        output_size = NUM_CLASSES

    # ✅ 입력 크기 설정(메타 우선, 동적 산출 제거)
    # 1) 기본은 메타 값(FEATURE_INPUT_SIZE)
    # 2) 호출자가 features를 넘기고, 그 차원이 메타보다 크면 그 값을 사용
    if input_size is None:
        input_size = FEATURE_INPUT_SIZE
        if features is not None and hasattr(features, "shape") and len(features.shape) >= 3:
            feat_dim = int(features.shape[2])
            if feat_dim >= FEATURE_INPUT_SIZE:
                input_size = feat_dim
                print(f"[info] input_size set from features: {input_size}")
            else:
                print(f"[info] features dim {feat_dim} < FEATURE_INPUT_SIZE {FEATURE_INPUT_SIZE} → use meta size")
        else:
            print(f"[info] input_size fixed to FEATURE_INPUT_SIZE={FEATURE_INPUT_SIZE}")

    # ✅ 입력 크기 검증/패드
    if input_size is None:
        raise ValueError("❌ get_model: input_size 계산 실패 → None 반환됨")
    if input_size < FEATURE_INPUT_SIZE:
        print(f"[info] input_size pad 적용: {input_size} → {FEATURE_INPUT_SIZE}")
        input_size = FEATURE_INPUT_SIZE

    # ✅ 모델 생성
    try:
        model = model_cls(input_size=input_size, output_size=output_size)
    except Exception as e:
        print(f"[⚠️ get_model 예외] {e}")
        print(f"[Fallback] input_size={FEATURE_INPUT_SIZE}로 재시도")
        model = model_cls(input_size=FEATURE_INPUT_SIZE, output_size=output_size)

    return model
