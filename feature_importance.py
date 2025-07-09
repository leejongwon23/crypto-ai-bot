import torch
import numpy as np
import os
import json
import pandas as pd

PERSIST_DIR = "/persistent"
IMPORTANCE_DIR = os.path.join(PERSIST_DIR, "importances")
os.makedirs(IMPORTANCE_DIR, exist_ok=True)

def compute_feature_importance(model, X_val, y_val, feature_names, method="baseline"):
    """
    ✅ feature importance 계산 (baseline permutation 방식)
    """
    model.eval()

    try:
        logits = model(X_val)
        y_val = y_val.view(-1).long()
        baseline_loss = torch.nn.CrossEntropyLoss()(logits, y_val).item()
    except Exception as e:
        print(f"[ERROR] 모델 예측 실패 ({method}): {e}")
        return dict(zip(feature_names, [0.0] * len(feature_names)))

    importances = []
    for i in range(X_val.shape[2]):
        try:
            X_permuted = X_val.clone()
            # ✅ permutation 방식 일관화
            perm_idx = torch.randperm(X_val.shape[0])
            X_permuted[:, :, i] = X_permuted[perm_idx, :, i]
            logits_perm = model(X_permuted)
            loss = torch.nn.CrossEntropyLoss()(logits_perm, y_val).item()
            importances.append(loss - baseline_loss)
        except Exception as e:
            print(f"[ERROR] 중요도 계산 실패 ({method} feature {i}): {e}")
            importances.append(0.0)

    return dict(zip(feature_names, importances))


def compute_feature_importance(model, X_val, y_val, feature_names, method="baseline"):
    """
    ✅ feature importance 계산 (baseline permutation 방식)
    """
    model.eval()

    try:
        logits = model(X_val)
        y_val = y_val.view(-1).long()
        baseline_loss = torch.nn.CrossEntropyLoss()(logits, y_val).item()
    except Exception as e:
        print(f"[ERROR] 모델 예측 실패 ({method}): {e}")
        return dict(zip(feature_names, [0.0] * len(feature_names)))

    importances = []
    for i in range(X_val.shape[2]):
        try:
            X_permuted = X_val.clone()
            # ✅ permutation 방식 일관화
            perm_idx = torch.randperm(X_val.shape[0])
            X_permuted[:, :, i] = X_permuted[perm_idx, :, i]
            logits_perm = model(X_permuted)
            loss = torch.nn.CrossEntropyLoss()(logits_perm, y_val).item()
            importances.append(loss - baseline_loss)
        except Exception as e:
            print(f"[ERROR] 중요도 계산 실패 ({method} feature {i}): {e}")
            importances.append(0.0)

    return dict(zip(feature_names, importances))


def compute_permutation_importance(model, X_val, y_val, feature_names):
    """
    ✅ 기존 permutation importance 함수 → compute_feature_importance와 동일 방식으로 통일
    """
    return compute_feature_importance(model, X_val, y_val, feature_names, method="permutation")

def save_feature_importance(importances, symbol, strategy, model_type, method="baseline"):
    suffix = f"_importance_{method}"
    fname_json = f"{symbol}_{strategy}_{model_type}{suffix}.json"
    fname_csv = f"{symbol}_{strategy}_{model_type}{suffix}.csv"
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
