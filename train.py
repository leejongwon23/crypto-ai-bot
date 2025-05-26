import os, json, torch, torch.nn as nn, numpy as np, datetime, pytz, sys, pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from data.utils import SYMBOLS, get_kline_by_strategy, compute_features
from model.base_model import get_model
from model_weight_loader import get_model_weight
from wrong_data_loader import load_wrong_prediction_data
from feature_importance import compute_feature_importance, save_feature_importance
import logger
from logger import get_min_gain, strategy_stats

DEVICE = torch.device("cpu")
DIR = "/persistent"; MODEL_DIR, LOG_DIR = f"{DIR}/models", f"{DIR}/logs"
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
now_kst = lambda: datetime.datetime.now(pytz.timezone("Asia/Seoul"))

def create_dataset(f, w):
    X, y = [], []
    for i in range(len(f) - w - 1):
        x_seq = f[i:i + w]
        if any(len(r.values()) != len(f[0].values()) for r in x_seq): continue
        c1, c2 = f[i + w - 1]['close'], f[i + w]['close']
        if c1 == 0: continue
        X.append([list(r.values()) for r in x_seq])
        y.append(round((c2 - c1) / c1, 4))
    if not X: return np.array([]), np.array([])
    mlen = max(set(map(len, X)), key=list(X).count)
    filt = [(x, l) for x, l in zip(X, y) if len(x) == mlen]
    return np.array([x for x, _ in filt]), np.array([l for _, l in filt]) if filt else (np.array([]), np.array([]))

def save_model_metadata(s, t, m, a, f1, l):
    meta = {"symbol": s, "strategy": t, "model": m, "accuracy": round(a,4), "f1_score": round(f1,4), "loss": round(l,6), "timestamp": now_kst().strftime("%Y-%m-%d %H:%M:%S")}
    path = f"{MODEL_DIR}/{s}_{t}_{m}.meta.json"
    with open(path, "w", encoding="utf-8") as f: json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"🗘 저장됨: {path}"); sys.stdout.flush()

def train_one_model(sym, strat, input_size=11, batch=32, epochs=10, lr=1e-3, rep=4, rep_wrong=4):
    print(f"[train] 🔄 {sym}-{strat} 시작"); sys.stdout.flush()
    try:
        win = find_best_window(sym, strat)
        df = get_kline_by_strategy(sym, strat)
        if df is None or len(df) < win + 10: raise ValueError("데이터 부족")
        df_feat = compute_features(sym, df, strat)
        if df_feat is None or len(df_feat) < win + 1: raise ValueError("feature 부족")
        feat = MinMaxScaler().fit_transform(df_feat.values)
        X_raw, y_raw = create_dataset([dict(zip(df_feat.columns, r)) for r in feat], win)
        if len(X_raw) < 2: raise ValueError("유효 시퀀스 부족")
        input_size, val_len = X_raw.shape[2], int(len(X_raw) * 0.2)
        if val_len == 0: raise ValueError("검증셋 부족")

        val_X, val_y = torch.tensor(X_raw[-val_len:], dtype=torch.float32), torch.tensor(y_raw[-val_len:], dtype=torch.float32)
        dataset = TensorDataset(torch.tensor(X_raw, dtype=torch.float32), torch.tensor(y_raw, dtype=torch.float32))
        train_set, _ = random_split(dataset, [len(dataset)-val_len, val_len])

        for model_type in ["lstm", "cnn_lstm", "transformer"]:
            model = get_model(model_type, input_size); model.train()
            optim, lossfn = torch.optim.Adam(model.parameters(), lr=lr), nn.MSELoss()
            loader = DataLoader(train_set, batch_size=batch, shuffle=True)

            try:
                # 🎯 학습 전 정확도 측정
                with torch.no_grad():
                    before_pred = model(val_X).squeeze(-1).numpy()
                    acc_before = r2_score(val_y.numpy(), before_pred)
            except:
                acc_before = ""

            for _ in range(rep):
                for _ in range(rep_wrong):
                    wrong_data = load_wrong_prediction_data(sym, strat, input_size, win)
                    if not wrong_data: continue
                    xb_all, yb_all = zip(*[(xb, yb) for xb, yb in wrong_data if xb.shape[1:] == (win, input_size)]) if wrong_data else ([],[])
                    if len(xb_all) >= 2:
                        xb, yb = torch.stack(xb_all), torch.tensor(yb_all, dtype=torch.float32)
                        for i in range(0, len(xb), batch):
                            rate = model(xb[i:i+batch]).squeeze(-1)
                            loss = lossfn(rate, yb[i:i+batch])
                            optim.zero_grad(); loss.backward(); optim.step()

                for xb, yb in loader:
                    rate = model(xb).squeeze(-1)
                    loss = lossfn(rate, yb)
                    optim.zero_grad(); loss.backward(); optim.step()

            model.eval()
            try:
                with torch.no_grad():
                    rate = model(val_X).squeeze(-1).numpy()
                    acc = r2_score(val_y.numpy(), rate)
                    f1 = mean_squared_error(val_y.numpy(), rate)
                    logloss = np.mean(np.square(val_y.numpy() - rate))
                    logger.log_training_result(sym, strat, model_type, acc, f1, logloss)
                    model_path = f"{MODEL_DIR}/{sym}_{strat}_{model_type}.pt"
                    torch.save(model.state_dict(), model_path)
                    print(f"✅ 저장: {model_path}"); sys.stdout.flush()
                    save_model_metadata(sym, strat, model_type, acc, f1, logloss)
                    imps = compute_feature_importance(model, val_X, val_y, list(df_feat.columns))
                    save_feature_importance(imps, sym, strat, model_type)

                    # 🎯 정확도 전후 기록 logger 연동
                    audit_row = {
                        "timestamp": now_kst().isoformat(),
                        "symbol": sym,
                        "strategy": strat,
                        "model": model_type,
                        "status": "train",
                        "reason": "train_complete",
                        "predicted_return": "",
                        "actual_return": "",
                        "accuracy_before": acc_before,
                        "accuracy_after": acc,
                        "predicted_volatility": "",
                        "actual_volatility": ""
                    }
                    audit_log_path = os.path.join(LOG_DIR, "evaluation_audit.csv")
                    write_header = not os.path.exists(audit_log_path) or os.stat(audit_log_path).st_size == 0
                    with open(audit_log_path, "a", newline="", encoding="utf-8-sig") as af:
                        writer = csv.DictWriter(af, fieldnames=list(audit_row.keys()))
                        if write_header: writer.writeheader()
                        writer.writerow(audit_row)

            except Exception as e:
                print(f"[평가 오류] {sym}-{strat}-{model_type} → {e}"); sys.stdout.flush()
    except Exception as e:
        print(f"[실패] {sym}-{strat} → {e}"); sys.stdout.flush()
        try:
            logger.log_training_result(sym, strat, f"실패({str(e)})", 0.0, 0.0, 0.0)
        except Exception as log_err:
            print(f"[로그 기록 실패] {sym}-{strat} → {log_err}"); sys.stdout.flush()

def train_all_models():
    strategy_order = ["단기", "중기", "장기"]
    def get_score(s):
        stat = strategy_stats.get(s, {"success": 0, "fail": 0, "returns": []})
        total = stat["success"] + stat["fail"]
        if total < 5: return 0
        success_rate = stat["success"] / total if total > 0 else 0
        avg_return = sum(stat["returns"]) / len(stat["returns"]) if stat["returns"] else 0
        return success_rate * avg_return
    strategy_order.sort(key=lambda s: -get_score(s))

    for strat in strategy_order:
        for sym in SYMBOLS:
            try: train_one_model(sym, strat)
            except Exception as e:
                print(f"[전체 학습 오류] {sym}-{strat} → {e}"); sys.stdout.flush()

def train_model_loop(strategy):
    for sym in SYMBOLS:
        try: train_one_model(sym, strategy)
        except Exception as e:
            print(f"[단일 학습 오류] {sym}-{strategy} → {e}"); sys.stdout.flush()

train_model = train_all_models
