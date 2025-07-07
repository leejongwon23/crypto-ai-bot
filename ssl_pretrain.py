import torch
import torch.nn as nn
import torch.optim as optim
from model.base_model import get_model
from data.utils import get_kline_by_strategy, compute_features
import numpy as np

DEVICE = torch.device("cpu")

def masked_reconstruction(symbol, strategy, input_size, mask_ratio=0.2, epochs=10):
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
    X = np.expand_dims(X, axis=0)  # (1, T, F)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    # ✅ transformer 모델로 reconstruction pretrain
    model = get_model("transformer", input_size=input_size, output_size=input_size).to(DEVICE)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lossfn = nn.MSELoss()

    for epoch in range(epochs):
        X_masked = X_tensor.clone()
        num_mask = int(X_masked.shape[1] * mask_ratio)
        mask_idx = np.random.choice(X_masked.shape[1], num_mask, replace=False)
        X_masked[:, mask_idx, :] = 0

        pred = model(X_masked)

        # 🔧 출력 크기 검사 및 reshape
        if pred.shape != X_tensor.shape:
            if pred.dim() == 3 and pred.shape[1] == 1:
                pred = pred.repeat(1, X_tensor.shape[1], 1)
            else:
                pred = pred.unsqueeze(1).repeat(1, X_tensor.shape[1], 1)

        loss = lossfn(pred, X_tensor)

        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[SSL] epoch {epoch+1}/{epochs}, loss={loss.item():.6f}")

    # ✅ pretraining weight 저장
    torch.save(model.state_dict(), f"/persistent/models/{symbol}_{strategy}_ssl.pt")
    print(f"[SSL] {symbol}-{strategy} pretraining 완료")
