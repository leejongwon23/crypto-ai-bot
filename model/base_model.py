# === model/base_model.py (speed-tune ready, fixed + output-size helper) ===
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import get_NUM_CLASSES, get_FEATURE_INPUT_SIZE

# ---- Optional deps (non-fatal) ----
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

FEATURE_INPUT_SIZE = get_FEATURE_INPUT_SIZE()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (옵션) 라벨/중립밴드 환경변수 — 학습 파이프라인에서 사용할 수 있음
LABEL_EPS = float(os.getenv("LABEL_EPS", "1e-9"))
FALLBACK_NEUTRAL_BAND = float(os.getenv("FALLBACK_NEUTRAL_BAND", "0.002"))

# ✅ 드롭아웃 하한(소폭 상향): 기본 0.2
GLOBAL_DROPOUT_MIN = float(os.getenv("MODEL_DROPOUT_MIN", "0.2"))
def _resolve_dropout(d):
    try:
        return max(float(d), GLOBAL_DROPOUT_MIN)
    except Exception:
        return GLOBAL_DROPOUT_MIN

# =========================
# Init Utilities (안정 초기화)
# =========================
def _init_module(m):
    if isinstance(m, nn.Linear):
        try:
            nn.init.xavier_uniform_(m.weight)
        except Exception:
            pass
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        try:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        except Exception:
            pass
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        if getattr(m, "weight", None) is not None:
            nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
        if getattr(m, "weight", None) is not None:
            nn.init.ones_(m.weight)

def _init_lstm_forget_bias(lstm: nn.LSTM):
    # LSTM 가중치/바이어스 안정 초기화 + forget gate bias = 1.0
    for name, param in lstm.named_parameters():
        if "weight_ih" in name:
            try:
                nn.init.xavier_uniform_(param)
            except Exception:
                pass
        elif "weight_hh" in name:
            try:
                nn.init.orthogonal_(param)
            except Exception:
                pass
        elif "bias" in name:
            try:
                nn.init.zeros_(param)
                n = param.shape[0] // 4
                param.data[n:2*n].fill_(1.0)  # forget gate bias
            except Exception:
                pass

# =========================
# TBPTT Mixin (훈련용 스텝)
# =========================
class TBPTTMixin:
    @torch.no_grad()
    def _detach_grads_(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()

    def tbptt_train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion,
        optimizer,
        chunk_len: int = 64,
        grad_clip: float = 1.0,
        amp: bool = False,
        scaler: "torch.cuda.amp.GradScaler | None" = None,
    ) -> float:
        """
        x: [B, T, F], y: [B] 분류 기준
        chunk별로 역전파와 step을 수행하여 메모리 사용을 제한.
        반환값: 평균 loss
        """
        self.train()
        T = int(x.size(1))
        chunks = max(1, (T + chunk_len - 1) // chunk_len)
        total = 0.0

        optimizer.zero_grad(set_to_none=True)

        for i in range(0, T, chunk_len):
            x_chunk = x[:, i:i + chunk_len, :]
            if x_chunk.numel() == 0:
                continue

            if amp and scaler is not None and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    logits = self.forward(x_chunk)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                logits = self.forward(x_chunk)
                loss = criterion(logits, y)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total += float(loss.detach().item())
            self._detach_grads_()  # 그래프 절단

        return total / float(chunks)

# =========================
# Attention (안정 소프트맥스 + dropout + 재정규화)
# =========================
class Attention(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)
        self.drop = nn.Dropout(_resolve_dropout(dropout))
        self._eps = 1e-12

    def forward(self, lstm_out):
        # lstm_out: [B, T, H]
        score = self.attn(lstm_out).squeeze(-1)                 # [B, T]
        # 안정화
        score = score - score.max(dim=1, keepdim=True).values
        weights = F.softmax(score, dim=1)                       # [B, T]
        # dropout on weights can break normalization -> apply then renormalize
        weights = self.drop(weights)
        s = weights.sum(dim=1, keepdim=True)
        weights = weights / (s + self._eps)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)  # [B, H]
        return context, weights

# =========================
# SE-like Context Gate (for LSTM/Transformer context)
# =========================
class ContextGate(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        r = max(1, dim // reduction)
        self.fc1 = nn.Linear(dim, r, bias=True)
        self.fc2 = nn.Linear(r, dim, bias=True)

    def forward(self, x):  # x: [B, H]
        z = F.gelu(self.fc1(x))
        g = torch.sigmoid(self.fc2(z))
        return x * g

# =========================
# 작은 ResConv 블록 (Conv-BN-ReLU-Conv-BN + Residual)
# =========================
class ResConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, p=1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in,  c_out, kernel_size=k, padding=p)
        self.bn1   = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=k, padding=p)
        self.bn2   = nn.BatchNorm1d(c_out)
        self.proj  = nn.Conv1d(c_in,  c_out, kernel_size=1) if c_in != c_out else nn.Identity()
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):  # [B,C,T]
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        residual = self.proj(x) if not isinstance(self.proj, nn.Identity) else x
        y = self.act(y + residual)
        return y

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
        z = F.gelu(self.fc1(s))
        z = torch.sigmoid(self.fc2(z))             # [B, C]
        return x * z.unsqueeze(-1)                 # [B, C, T]

# =========================
# LSTM Price Predictor (Residual Head + ContextGate)
# =========================
class LSTMPricePredictor(TBPTTMixin, nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.4, output_size=None):
        super().__init__()
        output_size = output_size if output_size is not None else get_NUM_CLASSES()
        H = hidden_size * 2
        dropout = _resolve_dropout(dropout)

        self.in_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = Attention(H, dropout=dropout * 0.5)
        self.ctx_norm = nn.LayerNorm(H)
        self.ctx_gate = ContextGate(H, reduction=16)

        self.dropout = nn.Dropout(dropout)
        # Residual head
        self.fc1 = nn.Linear(H, hidden_size)
        self.res_proj = nn.Linear(H, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_logits = nn.Linear(hidden_size // 2, output_size)

        _init_lstm_forget_bias(self.lstm)
        self.apply(_init_module)

    def forward(self, x, params=None):
        # x: [B, T, F]
        x = self.in_norm(x)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        context = self.ctx_norm(context)
        context = self.ctx_gate(context)
        context = self.dropout(context)

        if params is None:
            h1 = self.act(self.fc1(context))
            h1 = self.dropout(h1)
            h1 = h1 + self.res_proj(context)
            h2 = self.act(self.fc2(h1))
            h2 = self.dropout(h2)
            logits = self.fc_logits(h2)
        else:
            # meta-params path (weights provided externally)
            h1 = F.gelu(F.linear(context, params['fc1.weight'], params.get('fc1.bias')))
            h1 = F.dropout(h1, p=self.dropout.p, training=self.training)
            h1 = h1 + F.linear(context, params['res_proj.weight'], params.get('res_proj.bias'))
            h2 = F.gelu(F.linear(h1, params['fc2.weight'], params.get('fc2.bias')))
            h2 = F.dropout(h2, p=self.dropout.p, training=self.training)
            logits = F.linear(h2, params['fc_logits.weight'], params.get('fc_logits.bias'))
        return logits

# =========================
# CNN + LSTM Price Predictor (ResConv*2 + SE + BiLSTM + Attention + Residual Head)
# =========================
class CNNLSTMPricePredictor(TBPTTMixin, nn.Module):
    def __init__(self, input_size, cnn_channels=128, lstm_hidden_size=256, lstm_layers=3, dropout=0.4, output_size=None):
        super().__init__()
        output_size = output_size if output_size is not None else get_NUM_CLASSES()
        dropout = _resolve_dropout(dropout)

        self.in_norm = nn.LayerNorm(input_size)
        # conv expects channels -> we treat input_size as channel count after permute
        self.res1 = ResConvBlock(input_size,  cnn_channels, k=3, p=1)
        self.res2 = ResConvBlock(cnn_channels, cnn_channels, k=5, p=2)
        self.se    = SEBlock(cnn_channels, reduction=16)

        self.lstm = nn.LSTM(cnn_channels, lstm_hidden_size, lstm_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        H = lstm_hidden_size * 2
        self.attention = Attention(H, dropout=dropout * 0.5)
        self.ctx_norm = nn.LayerNorm(H)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(H, lstm_hidden_size)
        self.res_proj = nn.Linear(H, lstm_hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(lstm_hidden_size, lstm_hidden_size // 2)
        # ✅ 오타 수정: lmstm_hidden_size → lstm_hidden_size
        self.fc_logits = nn.Linear(lstm_hidden_size // 2, output_size)

        _init_lstm_forget_bias(self.lstm)
        self.apply(_init_module)

    def forward(self, x, params=None):
        # x: [B, T, F]
        x = self.in_norm(x)
        x_in = x.permute(0, 2, 1)           # [B, F, T]
        y = self.res1(x_in)
        y = self.res2(y)
        y = self.se(y)
        y = y.permute(0, 2, 1)              # [B, T, C]

        lstm_out, _ = self.lstm(y)
        context, _ = self.attention(lstm_out)
        context = self.ctx_norm(context)
        context = self.dropout(context)

        if params is None:
            h1 = self.act(self.fc1(context))
            h1 = self.dropout(h1)
            h1 = h1 + self.res_proj(context)
            h2 = self.act(self.fc2(h1))
            h2 = self.dropout(h2)
            logits = self.fc_logits(h2)
        else:
            h1 = F.gelu(F.linear(context, params['fc1.weight'], params.get('fc1.bias')))
            h1 = F.dropout(h1, p=self.dropout.p, training=self.training)
            h1 = h1 + F.linear(context, params['res_proj.weight'], params.get('res_proj.bias'))
            h2 = F.gelu(F.linear(h1, params['fc2.weight'], params.get('fc2.bias')))
            h2 = F.dropout(h2, p=self.dropout.p, training=self.training)
            logits = F.linear(h2, params['fc_logits.weight'], params.get('fc_logits.bias'))
        return logits

# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

# =========================
# Transformer Price Predictor (Residual Head)
# =========================
class TransformerPricePredictor(TBPTTMixin, nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dropout=0.4, output_size=None, mode="classification"):
        super().__init__()
        self.mode = mode
        output_size = output_size if output_size is not None else get_NUM_CLASSES()
        dropout = _resolve_dropout(dropout)

        self.in_norm = nn.LayerNorm(input_size)
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
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=512,
                    dropout=dropout, activation='gelu', batch_first=True
                )
            self.encoder_layers.append(layer)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if self.mode == "classification":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
            self.fc1 = nn.Linear(d_model, d_model // 2)
            self.res_proj = nn.Linear(d_model, d_model // 2)
            self.act = nn.GELU()
            self.fc_logits = nn.Linear(d_model // 2, output_size)
        elif self.mode == "reconstruction":
            self.decoder = nn.Linear(d_model, output_size)

        self.apply(_init_module)

    def forward(self, x, params=None):
        # x: [B, T, F]
        x = self.in_norm(x)
        x = self.input_proj(x)           # [B, T, D]
        if self.mode == "classification":
            B = x.size(0)
            cls = self.cls_token.expand(B, 1, -1)              # [B,1,D]
            x = torch.cat([cls, x], dim=1)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm(x)
        x = self.dropout(x)

        if self.mode == "classification":
            x_cls = x[:, 0]
            if params is None:
                h = self.act(self.fc1(x_cls))
                h = self.dropout(h)
                h = h + self.res_proj(x_cls)
                logits = self.fc_logits(h)
            else:
                h = F.gelu(F.linear(x_cls, params['fc1.weight'], params.get('fc1.bias')))
                h = F.dropout(h, p=self.dropout.p, training=self.training)
                h = h + F.linear(x_cls, params['res_proj.weight'], params.get('res_proj.bias'))
                logits = F.linear(h, params['fc_logits.weight'], params.get('fc_logits.bias'))
            return logits
        else:
            return self.decoder(x)

# =========================
# XGBoost Wrapper (optional)
# =========================
class XGBoostWrapper:
    def __init__(self, model_path):
        if not _HAS_XGB:
            raise RuntimeError("xgboost가 설치되어 있지 않습니다.")
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def predict(self, X):
        dmatrix = xgb.DMatrix(X.reshape(X.shape[0], -1))
        probs = self.model.predict(dmatrix)
        if isinstance(probs, np.ndarray):
            if probs.ndim == 1:
                try:
                    if probs.max() <= 1.0 and probs.min() >= 0.0:
                        return (probs > 0.5).astype(np.int64)
                    return (probs > 0).astype(np.int64)
                except Exception:
                    return np.argmax(np.vstack([1-probs, probs]).T, axis=1)
            elif probs.ndim == 2:
                return np.argmax(probs, axis=1)
        try:
            return np.argmax(np.array(probs), axis=1)
        except Exception:
            return np.asarray(probs).reshape(-1).astype(np.int64)

# =========================
# AutoEncoder (SSL/유틸)
# =========================
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, max(1, self.hidden_size // 2)),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(max(1, self.hidden_size // 2), self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.input_size)
        )
        self.apply(_init_module)

    def forward(self, x):
        # Accept x: [B, T, F] or [B, F] or [B,1,F]
        if x is None:
            return x
        if x.dim() == 3:
            B, T, F_ = x.shape
            if T == 1:
                flat = x.squeeze(1)
            else:
                flat = x.reshape(B, -1)
        elif x.dim() == 2:
            flat = x
        else:
            flat = x.view(x.size(0), -1)
        if flat.size(1) != self.input_size:
            if flat.size(1) > self.input_size:
                flat = flat[:, :self.input_size]
            else:
                pad = torch.zeros(flat.size(0), self.input_size - flat.size(1), device=flat.device, dtype=flat.dtype)
                flat = torch.cat([flat, pad], dim=1)
        encoded = self.encoder(flat)
        decoded = self.decoder(encoded)
        decoded = decoded.unsqueeze(1)
        return decoded

# =========================
# Registry & Factory
# =========================
MODEL_CLASSES = {
    "lstm": LSTMPricePredictor,
    "cnn_lstm": CNNLSTMPricePredictor,
    "cnn": CNNLSTMPricePredictor,          # 별칭 유지
    "transformer": TransformerPricePredictor,
    "xgboost": XGBoostWrapper,
    "autoencoder": AutoEncoder
}

def _normalize_model_type(mt: str) -> str:
    s = str(mt or "cnn_lstm").strip().lower()
    # 간단 별칭 정규화
    if s in ("xgb",): s = "xgboost"
    if s in ("cnn-lstm", "cnn+lstm"): s = "cnn_lstm"
    if s not in MODEL_CLASSES and s != "xgboost":
        # 알 수 없는 타입이면 기본 cnn_lstm
        s = "cnn_lstm"
    return s

def get_model(model_type="cnn_lstm", input_size=None, output_size=None,
              model_path=None, features=None, num_classes=None):
    """
    모델 객체를 생성하는 함수.
    train.py 에서 num_classes 로 전달해도 호환되도록 수정됨.
    """
    # 🔹 호환 처리: num_classes -> output_size 로 자동 대체
    if output_size is None and num_classes is not None:
        output_size = num_classes

    # 👇 이 호출에서 오는 output_size가 “이번 학습은 N클래스다”에 해당
    if output_size is None:
        output_size = get_NUM_CLASSES()

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

    if input_size < FEATURE_INPUT_SIZE:
        print(f"[info] input_size pad 적용: {input_size} → {FEATURE_INPUT_SIZE}")
        input_size = FEATURE_INPUT_SIZE

    mt = _normalize_model_type(model_type)

    if mt == "xgboost":
        if not _HAS_XGB or not model_path:
            print("[⚠️ get_model] XGBoost 사용 불가(미설치 또는 경로 없음). cnn_lstm 대체.")
            mt = "cnn_lstm"

    model_cls = MODEL_CLASSES.get(mt, CNNLSTMPricePredictor)

    try:
        if mt == "xgboost":
            model = model_cls(model_path=model_path)
        else:
            # 여기서 반드시 output_size로 마지막 헤드를 만든다
            model = model_cls(input_size=input_size, output_size=output_size)
    except Exception as e:
        print(f"[⚠️ get_model 예외] {e}")
        print(f"[Fallback] input_size={FEATURE_INPUT_SIZE}로 재시도")
        model = CNNLSTMPricePredictor(input_size=FEATURE_INPUT_SIZE, output_size=output_size)

    return model

# =========================
# ❗ 파인튜닝 유틸: 백본 동결/해제
# =========================
_HEADS_BY_TYPE = {
    LSTMPricePredictor:  ["fc_logits", "fc2", "fc1", "res_proj"],
    CNNLSTMPricePredictor: ["fc_logits", "fc2", "fc1", "res_proj"],
    TransformerPricePredictor: ["fc_logits", "fc1", "res_proj"]
}

def _mark_requires_grad(model: nn.Module, names_keep_on: set[str]):
    for n, p in model.named_parameters():
        p.requires_grad = (n.split('.')[0] in names_keep_on)

def freeze_backbone(model: nn.Module):
    """
    헤드만 학습하도록 백본을 모두 동결.
    """
    keep = set()
    for cls, heads in _HEADS_BY_TYPE.items():
        if isinstance(model, cls):
            keep = set(heads)
            break
    _mark_requires_grad(model, keep)
    return model

def unfreeze_last_k_layers(model: nn.Module, k: int = 1):
    """
    마지막 k개의 헤드 계층을 추가로 학습 가능하게 설정.
    """
    for cls, heads in _HEADS_BY_TYPE.items():
        if isinstance(model, cls):
            k = int(max(1, min(len(heads), k)))
            keep = set(heads[:k])  # heads는 [fc_logits, fc2, fc1, res_proj] 순
            _mark_requires_grad(model, keep)
            return model
    return model

# =========================
# 🔧 헤드 출력 크기 강제 맞추기 (predict 쪽에서 메타보고 재조정할 때 사용)
# =========================
def set_model_output_size(model: nn.Module, output_size: int) -> nn.Module:
    """
    이미 만들어진 모델의 마지막 분류 헤드를 주어진 output_size로 갈아끼운다.
    - 학습 시점에는 train.py가 get_model(..., output_size=N)으로 생성하므로 거의 안 쓰임
    - 예측 시점에 '이 모델은 12클래스였다'는 메타를 보고 맞춰야 할 때 사용
    """
    if not isinstance(output_size, int) or output_size <= 0:
        return model

    if isinstance(model, LSTMPricePredictor):
        in_f = model.fc_logits.in_features
        model.fc_logits = nn.Linear(in_f, output_size)
        _init_module(model.fc_logits)
    elif isinstance(model, CNNLSTMPricePredictor):
        in_f = model.fc_logits.in_features
        model.fc_logits = nn.Linear(in_f, output_size)
        _init_module(model.fc_logits)
    elif isinstance(model, TransformerPricePredictor) and model.mode == "classification":
        in_f = model.fc_logits.in_features
        model.fc_logits = nn.Linear(in_f, output_size)
        _init_module(model.fc_logits)
    # AutoEncoder, XGBoostWrapper 등은 여기서 안 건드림
    return model
