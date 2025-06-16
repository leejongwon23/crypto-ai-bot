import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from data.utils import get_kline_by_strategy, compute_features, create_dataset
from model.base_model import get_model

def find_best_window(symbol, strategy, window_list=[10, 20, 30, 40]):
    import os, json, torch
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from data.utils import get_kline_by_strategy, compute_features, create_dataset
    from model.base_model import get_model

    NUM_CLASSES = 21
    try:
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < max(window_list) + 20:
            print(f"[경고] {symbol}-{strategy} → 데이터 부족으로 기본값 반환")
            return 20

        df_feat = compute_features(symbol, df, strategy)
        if df_feat is None or df_feat.empty or df_feat.isnull().any().any():
            print(f"[경고] {symbol}-{strategy} → feature 부족 또는 NaN 포함으로 기본값 반환")
            return 20

        features = df_feat.to_dict(orient="records")

        best_acc = -1
        best_window = window_list[0]

        for window in window_list:
            try:
                X, y = create_dataset(features, window, strategy)
                if len(X) == 0 or len(y) == 0 or len(X) != len(y):
                    continue

                y = np.array(y)
                X = np.array(X)
                mask = (y >= 0) & (y < NUM_CLASSES)
                y = y[mask]
                X = X[mask]

                if len(X) < 10:
                    continue

                input_size = X.shape[2]
                model = get_model("lstm", input_size=input_size, output_size=NUM_CLASSES).train()

                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.long)

                val_len = max(5, int(len(X_tensor) * 0.2))
                train_X = X_tensor[:-val_len]
                train_y = y_tensor[:-val_len]
                val_X = X_tensor[-val_len:]
                val_y = y_tensor[-val_len:]

                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                lossfn = torch.nn.CrossEntropyLoss()

                for _ in range(5):
                    logits = model(train_X)
                    loss = lossfn(logits, train_y)
                    if not torch.isfinite(loss): break
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

                model.eval()
                with torch.no_grad():
                    logits_val = model(val_X)
                    preds = torch.argmax(logits_val, dim=1).numpy()
                    acc = accuracy_score(val_y.numpy(), preds)

                    if acc > best_acc:
                        best_acc = acc
                        best_window = window

            except Exception as e:
                print(f"[오류] window={window} 평가 실패 → {e}")
                continue

        print(f"[최적 WINDOW] {symbol}-{strategy} → {best_window} (acc: {best_acc:.4f})")
        return best_window

    except Exception as e:
        print(f"[find_best_window 오류] {symbol}-{strategy} → {e}")
        return 20
