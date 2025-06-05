# ✅ Render 캐시 강제 무효화용 주석 — 절대 삭제하지 마

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
    try:
        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < max(window_list) + 10:
            print(f"[경고] {symbol}-{strategy} → 데이터 부족으로 기본값 반환")
            return 20

        df_feat = compute_features(symbol, df, strategy)
        if df_feat is None or df_feat.empty or len(df_feat) < max(window_list) + 1:
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
                X, y = create_dataset(feature_dicts, window, strategy)
                if not isinstance(X, np.ndarray) or len(X) == 0 or len(X) != len(y):
                    print(f"[건너뜀] window={window}: X/y 길이 불일치 또는 데이터 없음")
                    continue
                if len(X.shape) != 3:
                    print(f"[건너뜀] window={window}: X 차원 오류 {X.shape}")
                    continue

                input_size = X.shape[2]
                model = get_model(model_type="lstm", input_size=input_size).train()

                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.long)

                val_len = int(len(X_tensor) * 0.2)
                train_len = len(X_tensor) - val_len
                if train_len <= 0 or val_len <= 0:
                    print(f"[건너뜀] window={window}: 훈련/검증 데이터 부족")
                    continue

                train_X = X_tensor[:train_len]
                train_y = y_tensor[:train_len]
                val_X = X_tensor[train_len:]
                val_y = y_tensor[train_len:]

                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                criterion = torch.nn.CrossEntropyLoss()

                for _ in range(3):
                    logits = model(train_X)
                    loss = criterion(logits, train_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

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
                print(f"[예외] window={window} 평가 실패 → {e}")
                continue

        # ✅ 저장
        save_dir = "/persistent/logs"
        os.makedirs(save_dir, exist_ok=True)
        save_txt = os.path.join(save_dir, f"best_window_{symbol}_{strategy}.txt")
        save_json = os.path.join(save_dir, f"best_window_{symbol}_{strategy}.json")

        try:
            with open(save_txt, "w") as f:
                f.write(str(best_window))
            with open(save_json, "w") as f:
                json.dump(best_result or {"window": best_window, "accuracy": 0}, f, indent=2)
        except Exception as e:
            print(f"[저장 오류] {symbol}-{strategy}: {e}")

        print(f"[최적 WINDOW] {symbol}-{strategy} → {best_window} (acc: {best_acc:.4f})")

        # ✅ 음수/비정상 방지
        return best_window if best_window > 0 else 20

    except Exception as e:
        print(f"[find_best_window 오류] {symbol}-{strategy} → {e}")
        return 20

