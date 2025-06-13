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
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score
    from data.utils import get_kline_by_strategy, compute_features, create_dataset
    from model.base_model import get_model

    try:
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < max(window_list) + 20:
            print(f"[경고] {symbol}-{strategy} → 데이터 부족으로 기본값 반환")
            return 20

        df_feat = compute_features(symbol, df, strategy)
        if df_feat is None or df_feat.empty or len(df_feat) < max(window_list) + 10:
            print(f"[경고] {symbol}-{strategy} → feature 부족으로 기본값 반환")
            return 20

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_feat.drop(columns=["timestamp"]).values)

        feature_dicts = []
        for i, row in enumerate(scaled):
            d = dict(zip(df_feat.columns.drop("timestamp"), row))
            d["timestamp"] = pd.to_datetime(df_feat.iloc[i]["timestamp"], errors="coerce")
            feature_dicts.append(d)

        best_acc = -1
        best_window = window_list[0]
        best_result = {}

        for window in window_list:
            try:
                result = create_dataset(feature_dicts, window, strategy)
                if not isinstance(result, (list, tuple)) or len(result) != 2:
                    continue

                X, y = result
                if X is None or y is None or len(X) == 0 or len(X) != len(y):
                    continue

                # ✅ 클래스 범위 초과 제거
                y = np.array(y)
                X = np.array(X)
                mask = (y >= 0) & (y < NUM_CLASSES)
                y = y[mask]
                X = X[mask]

                if len(X) == 0 or len(X) != len(y):
                    print(f"[스킵] window={window} → 유효한 샘플 없음")
                    continue

                input_size = X.shape[2]
                if input_size <= 0:
                    continue

                model = get_model("lstm", input_size, output_size=NUM_CLASSES).train()

                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.long)

                val_len = int(len(X_tensor) * 0.2)
                train_len = len(X_tensor) - val_len
                if train_len <= 0 or val_len <= 0:
                    continue

                train_X = X_tensor[:train_len]
                train_y = y_tensor[:train_len]
                val_X = X_tensor[train_len:]
                val_y = y_tensor[train_len:]

                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                criterion = torch.nn.CrossEntropyLoss()

                for _ in range(5):
                    logits = model(train_X)
                    loss = criterion(logits, train_y)
                    if not torch.isfinite(loss): break
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

                model.eval()
                with torch.no_grad():
                    logits_val = model(val_X)
                    preds = torch.argmax(logits_val, dim=1).numpy()
                    y_true = val_y.numpy()
                    acc = accuracy_score(y_true, preds)

                    if acc > best_acc:
                        best_acc = acc
                        best_window = window
                        best_result = {
                            "window": int(window),
                            "accuracy": float(round(acc, 4))
                        }

            except Exception as e:
                print(f"[오류] window={window} 평가 실패 → {e}")
                continue

        if best_acc < 0.0 or best_window not in window_list:
            print(f"[경고] {symbol}-{strategy}: 모든 창 평가 실패 → fallback=20")
            best_window = 20
            best_result = {"window": 20, "accuracy": 0.0}

        save_dir = "/persistent/logs"
        os.makedirs(save_dir, exist_ok=True)
        save_txt = os.path.join(save_dir, f"best_window_{symbol}_{strategy}.txt")
        save_json = os.path.join(save_dir, f"best_window_{symbol}_{strategy}.json")

        try:
            with open(save_txt, "w") as f:
                f.write(str(best_window))
            with open(save_json, "w") as f:
                json.dump(best_result, f, indent=2)
        except Exception as e:
            print(f"[저장 오류] {symbol}-{strategy}: {e}")

        print(f"[최적 WINDOW] {symbol}-{strategy} → {best_window} (acc: {best_acc:.4f})")
        return best_window

    except Exception as e:
        print(f"[find_best_window 오류] {symbol}-{strategy} → {e}")
        return 20

