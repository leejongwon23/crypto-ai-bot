# ssl_pretrain.py (FINAL — cache skip + unified path)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.base_model import TransformerPricePredictor
from data.utils import get_kline_by_strategy, compute_features
from config import get_FEATURE_INPUT_SIZE, get_SSL_CACHE_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_ssl_ckpt_path(symbol: str, strategy: str) -> str:
    """SSL 체크포인트 표준 경로"""
    ssl_dir = get_SSL_CACHE_DIR()
    os.makedirs(ssl_dir, exist_ok=True)
    return os.path.join(ssl_dir, f"{symbol}_{strategy}_ssl.pt")


def masked_reconstruction(
    symbol,
    strategy,
    input_size=None,
    mask_ratio=0.2,
    epochs=10,
    min_rows=50,
):
    """
    간단한 마스크드 재구성 사전학습.
    - input_size 미지정 시 config.get_FEATURE_INPUT_SIZE() 사용
    - 모델은 TransformerPricePredictor(mode='reconstruction')
    - ✅ 이미 학습된 ckpt가 있으면 '스킵'
    """
    ckpt_path = get_ssl_ckpt_path(symbol, strategy)
    if os.path.exists(ckpt_path):
        print(f"[SSL] cache found → skip: {ckpt_path}")
        return ckpt_path

    input_size = int(input_size or get_FEATURE_INPUT_SIZE())
    print(f"[SSL] {symbol}-{strategy} pretraining 시작 (input_size={input_size})")

    # --- 데이터 준비
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < min_rows:
        print(f"[SSL] 데이터 부족 → 스킵 (rows={len(df) if df is not None else 0}, 필요={min_rows})")
        return None

    feat = compute_features(symbol, df, strategy)
    if feat is None or feat.empty or feat.isnull().any().any():
        print("[SSL] feature 생성 실패 또는 NaN 포함 → 스킵")
        return None

    try:
        X = feat.drop(columns=["timestamp", "strategy"], errors="ignore").to_numpy(dtype=np.float32)
        X = np.expand_dims(X, axis=0)  # (1, T, F)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    except Exception as e:
        print(f"[SSL] 데이터 변환 실패 → 스킵 ({e})")
        return None

    # --- 모델/학습
    model = TransformerPricePredictor(
        input_size=input_size,
        output_size=input_size,
        mode="reconstruction"
    ).to(DEVICE)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lossfn = nn.MSELoss()

    num_mask = max(1, int(X_tensor.shape[1] * mask_ratio))
    num_mask = min(num_mask, max(1, X_tensor.shape[1] // 2))

    for epoch in range(epochs):
        X_masked = X_tensor.clone()
        mask_idx = np.random.choice(X_masked.shape[1], num_mask, replace=False)
        X_masked[:, mask_idx, :] = 0

        pred = model(X_masked)
        if pred.shape != X_tensor.shape:
            min_len = min(pred.shape[1], X_tensor.shape[1])
            pred = pred[:, :min_len, :]
            target = X_tensor[:, :min_len, :]
        else:
            target = X_tensor

        loss = lossfn(pred, target)
        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[SSL] epoch {epoch+1}/{epochs}, loss={loss.item():.6f}")

    # --- 저장(경로 통일)
    torch.save(model.state_dict(), ckpt_path)
    print(f"[SSL] {symbol}-{strategy} pretraining 완료 → 저장: {ckpt_path}")
    return ckpt_path
