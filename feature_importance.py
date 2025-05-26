import torch
import numpy as np
import os
import json
import pandas as pd

PERSIST_DIR = "/persistent"
IMPORTANCE_DIR = os.path.join(PERSIST_DIR, "importances")
os.makedirs(IMPORTANCE_DIR, exist_ok=True)

def compute_feature_importance(model, X_val, y_val, feature_names):
    model.eval()
    try:
        pred = model(X_val)
    except Exception as e:
        print(f"[ERROR] 모델 예측 실패 (기본): {e}")
        return dict(zip(feature_names, [0.0] * len(feature_names)))

    baseline_loss = torch.nn.MSELoss()(pred, y_val).item()
    importances = []
    for i in range(X_val.shape[2]):
        X_permuted = X_val.clone()
        X_permuted[:, :, i] = X_permuted[:, torch.randperm(X_val.shape[1]), i]
        try:
            perm_pred = model(X_permuted)
            loss = torch.nn.MSELoss()(perm_pred, y_val).item()
            importances.append(loss - baseline_loss)
        except Exception as e:
            print(f"[ERROR] 중요도 계산 실패 (feature {i}): {e}")
            importances.append(0.0)
    return dict(zip(feature_names, importances))

def compute_permutation_importance(model, X_val, y_val, feature_names):
    model.eval()
    try:
        pred = model(X_val)
    except Exception as e:
        print(f"[ERROR] 모델 예측 실패 (perm): {e}")
        return dict(zip(feature_names, [0.0] * len(feature_names)))

    baseline_loss = torch.nn.MSELoss()(pred, y_val).item()
    importances = []
    for i in range(X_val.shape[2]):
        X_permuted = X_val.clone()
        perm_idx = torch.randperm(X_val.shape[0])
        X_permuted[:, :, i] = X_permuted[perm_idx, :, i]
        try:
            perm_pred = model(X_permuted)
            loss = torch.nn.MSELoss()(perm_pred, y_val).item()
            importances.append(loss - baseline_loss)
        except Exception as e:
            print(f"[ERROR] 중요도 계산 실패 (perm feature {i}): {e}")
            importances.append(0.0)
    return dict(zip(feature_names, importances))

def save_feature_importance(importances, symbol, strategy, model_type):
    fname_json = f"{symbol}_{strategy}_{model_type}_importance.json"
    fname_csv = f"{symbol}_{strategy}_{model_type}_importance.csv"
    path_json = os.path.join(IMPORTANCE_DIR, fname_json)
    path_csv = os.path.join(IMPORTANCE_DIR, fname_csv)

    importances = {k: float(v) for k, v in importances.items()}

    with open(path_json, "w") as f:
        json.dump(importances, f, indent=2)

    df = pd.DataFrame(importances.items(), columns=["feature", "importance"]).sort_values(by="importance", ascending=False)
    df.to_csv(path_csv, index=False, encoding="utf-8-sig")

    print(f"✅ 중요도 저장 완료: {path_json}, {path_csv}")

def drop_low_importance_features(df: pd.DataFrame, importances: dict, threshold: float = 0.05) -> pd.DataFrame:
    drop_cols = [col for col, imp in importances.items() if imp < threshold]
    remaining_cols = [col for col in df.columns if col not in drop_cols]
    if not remaining_cols:
        print("[경고] 모든 feature가 제거되었음. 최소 1개 이상 유지 필요.")
        return df
    print(f"🧹 제거된 feature 수: {len(drop_cols)} → {drop_cols}")
    return df[remaining_cols]

def get_top_features(importances: dict, top_n: int = 10) -> pd.DataFrame:
    if not importances:
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.DataFrame(importances.items(), columns=["feature", "importance"])
    df_sorted = df.sort_values(by="importance", ascending=False).head(top_n)
    return df_sorted
