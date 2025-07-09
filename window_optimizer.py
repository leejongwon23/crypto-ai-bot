import os, json, torch, numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from data.utils import get_kline_by_strategy, compute_features, create_dataset
from model.base_model import get_model
from config import get_NUM_CLASSES
NUM_CLASSES = get_NUM_CLASSES()

def find_best_window(symbol, strategy, window_list=[10, 20, 30, 40]):
    from config import FEATURE_INPUT_SIZE, NUM_CLASSES  # ✅ config import 통일
    try:
        if not isinstance(window_list, list) or not all(isinstance(w, (int, float)) for w in window_list):
            print(f"[오류] window_list 타입 오류 → 기본값으로 대체")
            window_list = [10, 20, 30, 40]

        min_window = max(min(window_list), 5)

        df = get_kline_by_strategy(symbol, strategy)
        if df is None or len(df) < min(window_list) + 10:
            print(f"[경고] {symbol}-{strategy} → 데이터 부족으로 fallback window={min_window}")
            return min_window

        df_feat = compute_features(symbol, df, strategy)
        if df_feat is None or df_feat.empty or df_feat.isnull().any().any() or len(df_feat) < min(window_list) + 5:
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

        for window in sorted(window_list):
            if len(feature_dicts) <= window + 3:
                adjusted_window = max(5, len(feature_dicts) - 3)
                if adjusted_window < 5:
                    print(f"[⚠️ skip] window={window} (데이터 부족)")
                    continue
                print(f"[info] window={window} → 데이터 부족으로 adjusted_window={adjusted_window}")
                window = adjusted_window

            X, y = create_dataset(feature_dicts, window, strategy, input_size=FEATURE_INPUT_SIZE)
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
            if input_size != FEATURE_INPUT_SIZE:
                print(f"[⚠️ input_size 불일치] expected={FEATURE_INPUT_SIZE}, got={input_size} → 패딩 적용")
                pad_cols = FEATURE_INPUT_SIZE - input_size
                X = np.pad(X, ((0,0),(0,0),(0,pad_cols)), mode="constant", constant_values=0)
                input_size = FEATURE_INPUT_SIZE

            model = get_model("lstm", input_size, output_size=NUM_CLASSES).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = torch.nn.CrossEntropyLoss()

            val_len = max(1, int(len(X) * 0.2))
            if len(X) - val_len < 1:
                val_len = len(X) - 1

            train_X, val_X = X[:-val_len], X[-val_len:]
            train_y, val_y = y[:-val_len], y[-val_len:]

            if len(train_X) < 1 or len(val_X) < 1:
                print(f"[⚠️ skip] window={window} (train/val 샘플 부족)")
                continue

            for _ in range(3):
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
    ✅ 다중 윈도우 앙상블용 (개선)
    - feature 생성 후 유효성 체크
    - window_list 중 학습 가능한 window만 반환
    """
    from data.utils import compute_features, get_kline_by_strategy

    df = get_kline_by_strategy(symbol, strategy)
    if df is None or df.empty:
        print(f"[⚠️ find_best_windows] 데이터 없음 → 기본 window_list 반환")
        return window_list

    df_feat = compute_features(symbol, df, strategy)
    if df_feat is None or df_feat.empty or df_feat.isnull().any().any():
        print(f"[⚠️ find_best_windows] feature 생성 실패 또는 NaN → 기본 window_list 반환")
        return window_list

    valid_windows = []
    for w in window_list:
        if len(df_feat) >= w + 5:  # ✅ feature 기준으로 판단
            valid_windows.append(w)

    if not valid_windows:
        valid_windows = [min(window_list)]

    print(f"[✅ find_best_windows] {symbol}-{strategy} → {valid_windows}")
    return valid_windows


