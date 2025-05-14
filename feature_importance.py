import torch
import numpy as np
import os
import json

PERSIST_DIR = "/persistent"
IMPORTANCE_DIR = os.path.join(PERSIST_DIR, "importances")
os.makedirs(IMPORTANCE_DIR, exist_ok=True)

# --- 기존 LSTM용 중요도 (기존 그대로 유지)
def compute_feature_importance(model, X_val, y_val, feature_names):
    model.eval()
    baseline_loss = torch.nn.BCELoss()(model(X_val)[0], y_val).item()
    importances = []

    for i in range(X_val.shape[2]):
        X_permuted = X_val.clone()
        X_permuted[:, :, i] = X_permuted[:, torch.randperm(X_val.shape[1]), i]
        with torch.no_grad():
            loss = torch.nn.BCELoss()(model(X_permuted)[0], y_val).item()
        importances.append(loss - baseline_loss)

    return dict(zip(feature_names, importances))


# --- CNN_LSTM & Transformer용 permutation 중요도
def compute_permutation_importance(model, X_val, y_val, feature_names):
    model.eval()
    baseline_loss = torch.nn.BCELoss()(model(X_val)[0], y_val).item()
    importances = []

    for i in range(X_val.shape[2]):
        X_permuted = X_val.clone()
        perm_idx = torch.randperm(X_val.shape[0])
        X_permuted[:, :, i] = X_permuted[perm_idx, :, i]
        with torch.no_grad():
            loss = torch.nn.BCELoss()(model(X_permuted)[0], y_val).item()
        importances.append(loss - baseline_loss)

    return dict(zip(feature_names, importances))


def save_feature_importance(importances, symbol, strategy, model_type):
    fname = f"{symbol}_{strategy}_{model_type}_importance.json"
    path = os.path.join(IMPORTANCE_DIR, fname)
    with open(path, "w") as f:
        json.dump(importances, f, indent=2)
    print(f"✅ 중요도 저장됨: {path}")
