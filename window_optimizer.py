import os, json, torch, numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from data.utils import get_kline_by_strategy, compute_features, create_dataset
from model.base_model import get_model
from config import NUM_CLASSES

def find_best_window(symbol, strategy, window_list=[10, 20, 30, 40]):
    try:
        if not isinstance(window_list, list) or not all(isinstance(w, (int, float)) for w in window_list):
            print(f"[오류] window_list 타입 오류 → 기본값으로 대체")
            window_list = [10, 20, 30, 40]

        min_window = max(min(window_list), 10)

        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < max(window_list) + 20:
            print(f"[경고] {symbol}-{strategy} → 데이터 부족으로 fallback window={min_window}")
            return min_window

        df_feat = compute_features(symbol, df, strategy)
        if df_feat is None or df_feat.empty or df_feat.isnull().any().any() or len(df_feat) < max(window_list) + 10:
            print(f"[경고] {symbol}-{strategy} → feature 부족 또는 NaN 포함으로 fallback window={min_window}")
            return min_window

        drop_cols = ["timestamp"]
        if "strategy" in df_feat.columns:
            drop_cols.append("strategy")

        features_scaled = MinMaxScaler().fit_transform(df_feat.drop(columns=drop_cols))
        feature_dicts = []
        for i, row in enumerate(features_scaled):
            cols = df_feat.columns.drop(drop_cols)
            d = dict(zip(cols, row))
            d["timestamp"] = df_feat.iloc[i]["timestamp"]
            feature_dicts.append(d)

        best_acc = -1
        best_window = int(window_list[0])

        for window in window_list:
            if len(feature_dicts) <= window + 3:
                print(f"[⚠️ skip] window={window} (데이터 부족)")
                continue

            X, y = create_dataset(feature_dicts, window, strategy)
            if X is None or y is None or len(X) < 5:
                print(f"[⚠️ skip] window={window} (샘플 부족)")
                continue

            y = np.array(y)
            X = np.array(X)
            mask = (y >= 0) & (y < NUM_CLASSES)
            y = y[mask]
            X = X[mask]

            if len(X) < 5:
                continue

            input_size = X.shape[2]
            model = get_model("lstm", input_size, output_size=NUM_CLASSES).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = torch.nn.CrossEntropyLoss()

            val_len = max(5, int(len(X) * 0.2))
            train_X, val_X = X[:-val_len], X[-val_len:]
            train_y, val_y = y[:-val_len], y[-val_len:]

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

        if best_acc < 0.1:
            print(f"[경고] {symbol}-{strategy}: best_acc={best_acc:.4f} < 0.1 → fallback window={min_window}")
            best_window = int(min_window)

        print(f"[최적 WINDOW] {symbol}-{strategy} → {best_window} (acc: {best_acc:.4f})")
        return best_window

    except Exception as e:
        print(f"[find_best_window 오류] {symbol}-{strategy} → {e}")
        return min_window


def find_best_windows(symbol, strategy, window_list=[10, 20, 30, 40]):
    """
    ✅ 다중 윈도우 앙상블용
    - window_list 중 데이터 개수로 학습 가능한 window만 필터링
    - 반환: 유효한 window 리스트
    """
    df = get_kline_by_strategy(symbol, strategy)
    if df is None or df.empty:
        print(f"[⚠️ find_best_windows] 데이터 없음 → 기본 window_list 반환")
        return window_list

    valid_windows = []
    for w in window_list:
        if len(df) >= w + 10:
            valid_windows.append(w)
    if not valid_windows:
        valid_windows = [min(window_list)]
    print(f"[✅ find_best_windows] {symbol}-{strategy} → {valid_windows}")
    return valid_windows
