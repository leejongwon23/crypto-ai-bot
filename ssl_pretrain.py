import torch
import torch.nn as nn
import torch.optim as optim
from model.base_model import TransformerPricePredictor  # ✅ 직접 import
from data.utils import get_kline_by_strategy, compute_features
import numpy as np
from config import FEATURE_INPUT_SIZE
import os

DEVICE = torch.device("cpu")

def masked_reconstruction(symbol, strategy, input_size, mask_ratio=0.2, epochs=10):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from config import FEATURE_INPUT_SIZE
    from model.base_model import TransformerPricePredictor
    from data.utils import get_kline_by_strategy, compute_features

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[SSL] {symbol}-{strategy} pretraining 시작")

    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < 50:
        print("[SSL] 데이터 부족")
        return

    feat = compute_features(symbol, df, strategy)
    if feat is None or feat.empty:
        print("[SSL] feature 생성 실패")
        return

    X = feat.drop(columns=["timestamp", "strategy"], errors="ignore").values
    X = np.expand_dims(X, axis=0)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    model = TransformerPricePredictor(
        input_size=FEATURE_INPUT_SIZE,
        output_size=FEATURE_INPUT_SIZE,
        mode="reconstruction"
    ).to(DEVICE)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lossfn = nn.MSELoss()

    for epoch in range(epochs):
        X_masked = X_tensor.clone()
        num_mask = int(X_masked.shape[1] * mask_ratio)
        mask_idx = np.random.choice(X_masked.shape[1], num_mask, replace=False)
        X_masked[:, mask_idx, :] = 0

        pred = model(X_masked)

        if pred.shape != X_tensor.shape:
            print(f"[⚠️ shape mismatch] pred.shape={pred.shape}, target.shape={X_tensor.shape}")
            if pred.shape[1] != X_tensor.shape[1]:
                min_len = min(pred.shape[1], X_tensor.shape[1])
                pred = pred[:, :min_len, :]
                X_target = X_tensor[:, :min_len, :]
            else:
                X_target = X_tensor
        else:
            X_target = X_tensor

        loss = lossfn(pred, X_target)

        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[SSL] epoch {epoch+1}/{epochs}, loss={loss.item():.6f}")

    # ✅ 저장 경로 분리
    ssl_dir = "/persistent/ssl_models"
    os.makedirs(ssl_dir, exist_ok=True)
    model_path = f"{ssl_dir}/{symbol}_{strategy}_ssl.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[SSL] {symbol}-{strategy} pretraining 완료 → 저장: {model_path}")


