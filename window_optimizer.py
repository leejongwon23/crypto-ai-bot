# ✅ Render 캐시 강제 무효화용 주석 — 절대 삭제하지 마

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from data.utils import get_kline_by_strategy, compute_features
from model.base_model import get_model
import os, json

def create_dataset(features, window=20, strategy="단기"):
    # ✅ 전략별 예측 시간 설정
    horizon_map = {"단기": 4, "중기": 24, "장기": 168}
    target_hours = horizon_map.get(strategy, 4)

    for row in features:
        if isinstance(row.get("timestamp"), str):
            row["timestamp"] = pd.to_datetime(row["timestamp"])

    X, y = [], []
    for i in range(len(features) - window - 1):
        x_seq = features[i:i+window]
        base = features[i+window-1]
        base_time = base["timestamp"]
        base_price = base["close"]
        if base_price == 0:
            continue
        target_time = base_time + pd.Timedelta(hours=target_hours)
        future_slice = features[i+window:]
        target_row = next((r for r in future_slice if r["timestamp"] >= target_time), None)
        if not target_row:
            continue
        target_price = target_row["close"]
        X.append([list(r.values()) for r in x_seq])
        y.append(round((target_price - base_price) / base_price, 4))

    return np.array(X), np.array(y)


def find_best_window(symbol, strategy, window_list=[10, 20, 30, 40]):
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or len(df) < max(window_list) + 10:
        print(f"[경고] {symbol}-{strategy} → 데이터 부족으로 기본값 반환")
        return 20

    df_feat = compute_features(symbol, df, strategy)
    if df_feat is None or df_feat.empty or len(df_feat) < max(window_list) + 1:
        print(f"[경고] {symbol}-{strategy} → feature 부족으로 기본값 반환")
        return 20

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat.values)
    feature_dicts = [dict(zip(df_feat.columns, row)) for row in scaled]

    best_score = -1
    best_window = window_list[0]
    best_result = {}

    for window in window_list:
        X, y = create_dataset(feature_dicts, window, strategy)
        if len(X) == 0:
            continue

        input_size = X.shape[2] if len(X.shape) == 3 else X.shape[1]
        model = get_model(model_type="lstm", input_size=input_size)
        model.train()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        val_len = int(len(X_tensor) * 0.2)
        train_len = len(X_tensor) - val_len
        if train_len <= 0 or val_len <= 0:
            continue
        train_X = X_tensor[:train_len]
        train_y = y_tensor[:train_len]
        val_X = X_tensor[train_len:]
        val_y = y_tensor[train_len:]

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        try:
            for _ in range(3):
                pred = model(train_X).squeeze()
                loss = criterion(pred, train_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_val = model(val_X).squeeze().numpy()
                score = r2_score(val_y.numpy(), pred_val)

                if score > best_score:
                    best_score = score
                    best_window = window
                    best_result = {
                        "window": int(window),
                        "r2_score": float(round(score, 4))
                    }

        except Exception as e:
            print(f"[오류] window={window} 평가 실패 → {e}")
            continue

    save_dir = "/persistent/logs"
    os.makedirs(save_dir, exist_ok=True)
    save_txt = os.path.join(save_dir, f"best_window_{symbol}_{strategy}.txt")
    save_json = os.path.join(save_dir, f"best_window_{symbol}_{strategy}.json")

    try:
        with open(save_txt, "w") as f:
            f.write(str(best_window))
        with open(save_json, "w") as f:
            json.dump(best_result or {"window": best_window, "r2_score": 0}, f, indent=2)
    except Exception as e:
        print(f"[저장 오류] {symbol}-{strategy}: {e}")

    print(f"[최적 WINDOW] {symbol}-{strategy} → {best_window} (r2_score: {best_score:.4f})")
    return best_window
