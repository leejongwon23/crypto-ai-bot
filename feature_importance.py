import torch
import numpy as np
import os
import json

PERSIST_DIR = "/persistent"
IMPORTANCE_DIR = os.path.join(PERSIST_DIR, "importances")
os.makedirs(IMPORTANCE_DIR, exist_ok=True)

# --- 중요도 분석 (모델 출력이 튜플인지 안전 체크 포함)
def compute_feature_importance(model, X_val, y_val, feature_names):
    model.eval()

    try:
        pred, *_ = model(X_val)
    except Exception as e:
        print(f"[ERROR] 모델 예측 실패 (기본): {e}")
        return dict(zip(feature_names, [0.0] * len(feature_names)))

    baseline_loss = torch.nn.BCELoss()(pred, y_val).item()
    importances = []

    for i in range(X_val.shape[2]):
        X_permuted = X_val.clone()
        X_permuted[:, :, i] = X_permuted[:, torch.randperm(X_val.shape[1]), i]
        try:
            perm_pred, *_ = model(X_permuted)
            loss = torch.nn.BCELoss()(perm_pred, y_val).item()
            importances.append(loss - baseline_loss)
        except Exception as e:
            print(f"[ERROR] 중요도 계산 실패 (feature {i}): {e}")
            importances.append(0.0)

    return dict(zip(feature_names, importances))

# --- CNN_LSTM & Transformer용 permutation 중요도 (예외 처리 포함)
def compute_permutation_importance(model, X_val, y_val, feature_names):
    model.eval()

    try:
        pred, *_ = model(X_val)
    except Exception as e:
        print(f"[ERROR] 모델 예측 실패 (perm): {e}")
        return dict(zip(feature_names, [0.0] * len(feature_names)))

    baseline_loss = torch.nn.BCELoss()(pred, y_val).item()
    importances = []

    for i in range(X_val.shape[2]):
        X_permuted = X_val.clone()
        perm_idx = torch.randperm(X_val.shape[0])
        X_permuted[:, :, i] = X_permuted[perm_idx, :, i]
        try:
            perm_pred, *_ = model(X_permuted)
            loss = torch.nn.BCELoss()(perm_pred, y_val).item()
            importances.append(loss - baseline_loss)
        except Exception as e:
            print(f"[ERROR] 중요도 계산 실패 (perm feature {i}): {e}")
            importances.append(0.0)

    return dict(zip(feature_names, importances))

def save_feature_importance(importances, symbol, strategy, model_type):
    fname = f"{symbol}_{strategy}_{model_type}_importance.json"
    path = os.path.join(IMPORTANCE_DIR, fname)
    with open(path, "w") as f:
        json.dump(importances, f, indent=2)
    print(f"✅ 중요도 저장됨: {path}")
