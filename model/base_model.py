import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import numpy as np
from config import get_NUM_CLASSES, get_FEATURE_INPUT_SIZE

NUM_CLASSES = get_NUM_CLASSES()
FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Init Utilities (안정 초기화)
# =========================
def _init_module(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def _init_lstm_forget_bias(lstm: nn.LSTM, hidden_size: int):
    # LSTM 가중치/바이어스 안정 초기화 + forget gate bias = 1.0
    for name, param in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
            # i, f, g, o 게이트 순서 가정(pytorch 표준)
            n = param.shape[0] // 4
            param.data[n:2*n].fill_(1.0)

# =========================
# Attention (안정 소프트맥스 + dropout)
# =========================
class Attention(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, lstm_out):
        # lstm_out: [B, T, H]
        score = self.attn(lstm_out).squeeze(-1)                 # [B, T]
        score = score - score.max(dim=1, keepdim=True).values   # 수치 안정화
        weights = F.softmax(score, dim=1)                       # [B, T]
        weights = self.drop(weights)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)  # [B, H]
        return context, weights

# =========================
# LSTM Price Predictor
# =========================
class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.4, output_size=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2, dropout=dropout * 0.5)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_logits = nn.Linear(hidden_size // 2, output_size)

        # 안정 초기화
        _init_lstm_forget_bias(self.lstm, hidden_size)
        self.apply(_init_module)

    def forward(self, x, params=None):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.norm(context)
        context = self.dropout(context)

        if params is None:
            hidden = self.act(self.fc1(context))
            hidden = self.dropout(hidden)
            hidden = self.act(self.fc2(hidden))
            hidden = self.dropout(hidden)
            logits = self.fc_logits(hidden)
        else:
            hidden = F.gelu(F.linear(context, params['fc1.weight'], params['fc1.bias']))
            hidden = F.dropout(hidden, p=self.dropout.p, training=self.training)
            hidden = F.gelu(F.linear(hidden, params['fc2.weight'], params['fc2.bias']))
            hidden = F.dropout(hidden, p=self.dropout.p, training=self.training)
            logits = F.linear(hidden, params['fc_logits.weight'], params['fc_logits.bias'])
        return logits

# =========================
# Squeeze-and-Excitation Block (채널 주의집중)
# =========================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, r, bias=True)
        self.fc2 = nn.Linear(r, channels, bias=True)

    def forward(self, x):  # x: [B, C, T]
        s = x.mean(dim=2)                          # [B, C]
        z = F.relu(self.fc1(s))
        z = torch.sigmoid(self.fc2(z))             # [B, C]
        return x * z.unsqueeze(-1)                 # [B, C, T]

# =========================
# CNN + LSTM Price Predictor
# =========================
class CNNLSTMPricePredictor(nn.Module):
    def __init__(self, input_size, cnn_channels=128, lstm_hidden_size=256, lstm_layers=3, dropout=0.4, output_size=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(cnn_channels)
        self.se    = SEBlock(cnn_channels, reduction=16)

        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden_size, lstm_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = Attention(lstm_hidden_size * 2, dropout=dropout * 0.5)
        self.norm = nn.LayerNorm(lstm_hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_size * 2, lstm_hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(lstm_hidden_size, lstm_hidden_size // 2)
        self.fc_logits = nn.Linear(lstm_hidden_size // 2, output_size)

        # 안정 초기화
        _init_lstm_forget_bias(self.lstm, lstm_hidden_size)
        self.apply(_init_module)

    def forward(self, x, params=None):
        # CNN feature extractor
        x = x.permute(0, 2, 1)           # [B, F, T]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        x = x.permute(0, 2, 1)           # [B, T, C]

        # LSTM + Attention
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.norm(context)
        context = self.dropout(context)

        if params is None:
            hidden = self.act(self.fc1(context))
            hidden = self.dropout(hidden)
            hidden = self.act(self.fc2(hidden))
            hidden = self.dropout(hidden)
            logits = self.fc_logits(hidden)
        else:
            hidden = F.gelu(F.linear(context, params['fc1.weight'], params['fc1.bias']))
            hidden = F.dropout(hidden, p=self.dropout.p, training=self.training)
            hidden = F.gelu(F.linear(hidden, params['fc2.weight'], params['fc2.bias']))
            hidden = F.dropout(hidden, p=self.dropout.p, training=self.training)
            logits = F.linear(hidden, params['fc_logits.weight'], params['fc_logits.bias'])
        return logits

# =========================
# Transformer Price Predictor
# =========================
class TransformerPricePredictor(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dropout=0.4, output_size=None, mode="classification"):
        super().__init__()
        self.mode = mode
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            try:
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=512,
                    dropout=dropout, activation='gelu', batch_first=True, norm_first=True
                )
            except TypeError:
                # older torch: norm_first 없음
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=512,
                    dropout=dropout, activation='gelu', batch_first=True
                )
            self.encoder_layers.append(layer)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 분류 헤드
        if self.mode == "classification":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
            self.fc1 = nn.Linear(d_model, d_model // 2)
            self.act = nn.GELU()
            self.fc_logits = nn.Linear(d_model // 2, output_size)
        elif self.mode == "reconstruction":
            self.decoder = nn.Linear(d_model, output_size)

        # 안정 초기화
        self.apply(_init_module)

    def forward(self, x, params=None):
        x = self.input_proj(x)           # [B, T, D]
        if self.mode == "classification":
            B = x.size(0)
            cls = self.cls_token.expand(B, 1, -1)              # [B,1,D]
            x = torch.cat([cls, x], dim=1)                     # CLS prepend
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm(x)
        x = self.dropout(x)

        if self.mode == "classification":
            x = x[:, 0]  # CLS 토큰만
            if params is None:
                x = self.act(self.fc1(x))
                x = self.dropout(x)
                logits = self.fc_logits(x)
            else:
                x = F.gelu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
                x = F.dropout(x, p=self.dropout.p, training=self.training)
                logits = F.linear(x, params['fc_logits.weight'], params['fc_logits.bias'])
            return logits
        elif self.mode == "reconstruction":
            return self.decoder(x)

# =========================
# Positional Encoding
# =========================
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

# =========================
# XGBoost Wrapper (비활성 안내 그대로)
# =========================
class XGBoostWrapper:
    def __init__(self, model_path):
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def predict(self, X):
        dmatrix = xgb.DMatrix(X.reshape(X.shape[0], -1))
        probs = self.model.predict(dmatrix)
        return np.argmax(probs, axis=1)

# =========================
# AutoEncoder (SSL/유틸)
# =========================
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
        x = x.squeeze(1)           # [B, F]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.unsqueeze(1)
        return decoded

# =========================
# Registry & Factory
# =========================
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

    # ✅ MODEL_CLASSES 직접 사용
    if model_type not in MODEL_CLASSES:
        print(f"[경고] 알 수 없는 모델 타입 '{model_type}', 기본 모델 cnn_lstm 사용")
        model_cls = CNNLSTMPricePredictor
    else:
        model_cls = MODEL_CLASSES[model_type]

    # ✅ 출력 클래스 수 설정
    if output_size is None:
        output_size = NUM_CLASSES

    # ✅ 입력 크기 설정(메타 우선, features가 더 크면 승격)
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
