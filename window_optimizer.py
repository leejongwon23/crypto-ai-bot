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
    import os, json
    import torch
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
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
        if df_feat is None or df_feat.empty or df_feat.isnull().any().any() or len(df_feat) < max(window_list) + 10:
            print(f"[경고] {symbol}-{strategy} → feature 부족 또는 NaN 포함으로 기본값 반환")
            return 20

        # ✅ 구조 통일: 기존 방식과 동일하게 dict 리스트로 변환
        feature_dicts = df_feat.to_dict(orient="records")

        best_acc = -1
        best_window = window_list[0]
        best_result = {}

        for window in window_list:
            try:
                X, y = create_dataset(feature_dicts, window, strategy)
                if X is None or y is None or len(X) == 0 or len(X) != len(y):
                    continue

                y = np.array(y)
                X = np.array(X)
                mask = (y >= 0) & (y < NUM_CLASSES)
                y = y[mask]
                X = X[mask]

                if len(X) == 0 or len(X) != len(y):
                    continue

                input_size = X.shape[2]
                model = get_model("lstm", input_size, output_size=NUM_CLASSES).train()

                val_len = max(5, int(len(X) * 0.2))
                train_X, val_X = X[:-val_len], X[-val_len:]
                train_y, val_y = y[:-val_len], y[-val_len:]

                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                loss_fn = torch.nn.CrossEntropyLoss()

                for _ in range(5):
                    model.train()
                    logits = model(torch.tensor(train_X, dtype=torch.float32))
                    loss = loss_fn(logits, torch.tensor(train_y, dtype=torch.long))
                    if not torch.isfinite(loss): break
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

                model.eval()
                with torch.no_grad():
                    logits_val = model(torch.tensor(val_X, dtype=torch.float32))
                    preds = torch.argmax(logits_val, dim=1).numpy()
                    acc = accuracy_score(val_y, preds)

                    if acc > best_acc:
                        best_acc = acc
                        best_window = window
                        best_result = {"window": int(window), "accuracy": float(round(acc, 4))}

            except Exception as e:
                print(f"[오류] window={window} 평가 실패 → {e}")
                continue

        if best_acc < 0.0 or best_window not in window_list:
            print(f"[경고] {symbol}-{strategy}: 모든 창 평가 실패 → fallback=20")
            best_window = 20
            best_result = {"window": 20, "accuracy": 0.0}

        os.makedirs("/persistent/logs", exist_ok=True)
        with open(f"/persistent/logs/best_window_{symbol}_{strategy}.txt", "w") as f:
            f.write(str(best_window))
        with open(f"/persistent/logs/best_window_{symbol}_{strategy}.json", "w") as f:
            json.dump(best_result, f, indent=2)

        print(f"[최적 WINDOW] {symbol}-{strategy} → {best_window} (acc: {best_acc:.4f})")
        return best_window

    except Exception as e:
        print(f"[find_best_window 오류] {symbol}-{strategy} → {e}")
        return 20

    
