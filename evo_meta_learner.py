# evo_meta_learner.py (PATCHED: input/DataFrame fallback + torch safety + save/load robustness + predict input shape guard)
import os
import json
import ast
import numpy as np
import pandas as pd

# torch optional safe import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None

MODEL_PATH = "/persistent/models/evo_meta_learner.pt"
META_PATH  = MODEL_PATH.replace(".pt", ".meta.json")

class EvoMetaModel(nn.Module if nn is not None else object):
    def __init__(self, input_size, hidden_size=64, output_size=3):
        if nn is None:
            return
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        if F is None:
            return x
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def _safe_lit(x, default):
    try:
        if x is None or str(x).strip() == "":
            return default
        return ast.literal_eval(str(x))
    except Exception:
        return default

def _df_from_path_or_df(path_or_df):
    """
    Accept either a path string or a pandas DataFrame.
    """
    if path_or_df is None:
        return None
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df
    if isinstance(path_or_df, str):
        if not os.path.exists(path_or_df):
            return None
        try:
            return pd.read_csv(path_or_df, encoding="utf-8-sig")
        except Exception:
            try:
                return pd.read_csv(path_or_df)
            except Exception:
                return None
    return None

def prepare_evo_meta_dataset(path_or_df="/persistent/wrong_predictions.csv", min_samples=50):
    """
    실패/오답 로그를 기반으로 '전략 선택' 학습용 피처 구성.
    Accepts either CSV path or already-loaded DataFrame.
    """
    df = _df_from_path_or_df(path_or_df)
    if df is None:
        print(f"[❌ prepare_evo_meta_dataset] 파일/데이터 없음 또는 읽기 실패: {path_or_df}")
        return None, None

    try:
        if len(df) < min_samples:
            print(f"[❌ prepare_evo_meta_dataset] 샘플 부족: {len(df)}개 (min={min_samples})")
            return None, None
    except Exception:
        print("[❌ prepare_evo_meta_dataset] 데이터 길이 확인 실패")
        return None, None

    X_list, y_list = [], []
    for _, row in df.iterrows():
        try:
            sm = _safe_lit(row.get("softmax"), [])
            er = _safe_lit(row.get("expected_returns"), [0, 0, 0])
            mp = _safe_lit(row.get("model_predictions"), [0, 0, 0])
            # require softmax-like vector length 3
            if not sm or len(sm) < 3:
                continue
            # label may be missing; best to proceed with safe defaults
            label = int(float(row.get("label", -1))) if pd.notnull(row.get("label")) else -1
            feats = []
            for i in range(3):
                s_val = float(sm[i]) if i < len(sm) else 0.0
                e_val = float(er[i]) if i < len(er) else 0.0
                hit = 1 if (i < len(mp) and int(mp[i]) == label and label >= 0) else 0
                feats.extend([s_val, e_val, hit])
            # ensure feature vector length = 9
            if len(feats) != 9:
                continue
            X_list.append(feats)
            # best_strategy fallback
            best_str = row.get("best_strategy", 0)
            try:
                y_list.append(int(float(best_str)))
            except Exception:
                y_list.append(0)
        except Exception:
            continue

    if not X_list or not y_list:
        print("[❌ prepare_evo_meta_dataset] 유효 샘플 부족")
        return None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"[✅ prepare_evo_meta_dataset] X:{X.shape}, y:{y.shape}")
    return X, y

def train_evo_meta(X, y, input_size, output_size=3, epochs=10, batch_size=32, lr=1e-3, task="strategy"):
    """
    task: "strategy"(기본) 또는 "class" — 메타 JSON에 기록되어 추론 시 가드로 사용.
    """
    if torch is None:
        raise RuntimeError("torch is required for train_evo_meta")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EvoMetaModel(input_size, output_size=output_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"[evo_meta_learner] 학습 시작 → 샘플:{len(dataset)}, output_size:{output_size}, input:{input_size}, task={task}")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += float(loss.item())
        print(f"[evo_meta_learner] Epoch {epoch+1}/{epochs} → Loss: {epoch_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        # save cpu state dict to avoid device-specific tensors
        model_cpu = model.to("cpu")
        torch.save(model_cpu.state_dict(), MODEL_PATH)
    finally:
        # return model to DEVICE if needed
        try:
            model.to(DEVICE)
        except Exception:
            pass

    meta_info = {
        "task": str(task),               # "strategy" 또는 "class"
        "input_size": int(input_size),
        "output_size": int(output_size),
        "version": 1
    }
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] meta save 실패: {e}")

    print(f"[✅ evo_meta_learner] 학습 완료 → {MODEL_PATH} (task={task}, output_size={output_size})")
    return model

def train_evo_meta_loop(min_samples=50, auto_trigger=False, task="strategy", path_or_df="/persistent/wrong_predictions.csv"):
    """
    기본은 '전략 선택' 학습. (task='strategy')
    """
    if task == "strategy":
        X, y = prepare_evo_meta_dataset(path_or_df, min_samples=min_samples)
    else:
        X, y = None, None

    if X is None or y is None:
        if auto_trigger:
            print("[⏭️ evo_meta_learner] 데이터 부족 → 자동 학습 스킵")
        return

    input_size = int(X.shape[1])
    output_size = int(len(np.unique(y))) if len(np.unique(y)) > 0 else 3
    train_evo_meta(X, y, input_size=input_size, output_size=output_size, task=task)
    print("[✅ evo_meta_learner] 학습 완료 및 모델 저장됨")

def _load_meta():
    meta = {}
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass
    return meta

def predict_evo_meta(X_new, input_size):
    """
    predict.py에서 불러 쓰는 진입점.
    현재 저장된 메타가 task!='class'면 '클래스 선택' 용도가 아니므로 None 반환(안전 가드).
    """
    if torch is None:
        print("[❌ evo_meta_learner] torch 미설치")
        return None

    if not os.path.exists(MODEL_PATH):
        print("[❌ evo_meta_learner] 모델 없음")
        return None

    meta = _load_meta()
    task = meta.get("task", "class")
    if task != "class":
        print(f"[ℹ️ evo_meta_learner] task={task} → class 예측에 사용하지 않음")
        return None

    out_size = int(meta.get("output_size", 3))
    model = EvoMetaModel(input_size, output_size=out_size)
    try:
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"[❌ evo_meta_learner] 모델 로드 실패: {e}")
        return None

    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_new, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        try:
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            best = int(torch.argmax(probs, dim=1).item())
            return best
        except Exception as e:
            print(f"[❌ evo_meta_learner] 예측 실패: {e}")
            return None

def get_best_strategy_by_failure_probability(symbol, current_strategy, feature_tensor, model_outputs):
    """
    간단 휴리스틱: 최근 심볼별 전략 성공률을 보고 제안.
    """
    try:
        path = "/persistent/prediction_log.csv"
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        df = df[df["symbol"] == symbol]
        df = df[df["status"].isin(["success","fail","v_success","v_fail"])]
        if df.empty:
            return None
        sr = df.pivot_table(index="strategy", values="status",
                            aggfunc=lambda s: (s.isin(["success","v_success"])).mean())
        sr = sr["status"].to_dict()
        curr = float(sr.get(current_strategy, 0.0))
        if curr < 0.25:
            alt = sorted([(k, float(v)) for k, v in sr.items() if k != current_strategy],
                         key=lambda x: x[1], reverse=True)
            if alt and alt[0][1] >= 0.45:
                return alt[0][0]
        return None
    except Exception:
        return None
