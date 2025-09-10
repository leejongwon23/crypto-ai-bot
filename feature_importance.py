# === feature_importance.py (SAFE: no_grad, 시간예산/조기중단, 퍼뮤테이션 통일) ===
import os
import json
import time
import torch
import numpy as np
import pandas as pd

PERSIST_DIR = "/persistent"
IMPORTANCE_DIR = os.path.join(PERSIST_DIR, "importances")
os.makedirs(IMPORTANCE_DIR, exist_ok=True)

def _ensure_feature_names(feature_names, n_features):
    if isinstance(feature_names, (list, tuple)) and len(feature_names) == n_features:
        return list(feature_names)
    # 부족/없음 → 기본 이름 부여
    return [f"f_{i}" for i in range(n_features)]

@torch.no_grad()
def compute_feature_importance(
    model,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    feature_names,
    method: str = "baseline",
    *,
    max_seconds: float = 30.0,    # ⏱️ 전체 계산 시간예산(초)
    print_every: int = 20         # 진행 로그 간격(특성 개수 기준)
):
    """
    ✅ feature importance 계산 (permutation 방식, 메모리/시간 가드)
    - 입력 X_val shape: (N, T, F)
    - 예산 초과 시 남은 피처는 0.0으로 채우고 조기 종료
    """
    model.eval()
    start = time.time()

    try:
        logits = model(X_val)
        y_val = y_val.view(-1).long()
        baseline_loss = torch.nn.CrossEntropyLoss()(logits, y_val).item()
    except Exception as e:
        print(f"[ERROR] 모델 예측 실패 ({method}): {e}")
        n_feat = X_val.shape[2]
        names = _ensure_feature_names(feature_names, n_feat)
        return dict(zip(names, [0.0] * n_feat))

    n_feat = X_val.shape[2]
    names = _ensure_feature_names(feature_names, n_feat)
    importances = np.zeros(n_feat, dtype=float)

    for i in range(n_feat):
        # ⏱️ 시간 예산 체크
        if (time.time() - start) > float(max_seconds):
            print(f"[WARN] importance 시간예산 초과 → i={i}/{n_feat}에서 중단(잔여=0.0)")
            break
        try:
            X_perm = X_val.clone()
            # ✅ permutation 일관화: 배치 축만 셔플
            perm_idx = torch.randperm(X_val.shape[0], device=X_val.device)
            X_perm[:, :, i] = X_perm[perm_idx, :, i]
            logits_perm = model(X_perm)
            loss = torch.nn.CrossEntropyLoss()(logits_perm, y_val).item()
            importances[i] = float(loss - baseline_loss)
            if print_every and (i % max(1, int(print_every)) == 0):
                print(f"[imp] {i+1}/{n_feat} lossΔ={importances[i]:.6f}")
        except Exception as e:
            print(f"[ERROR] 중요도 계산 실패 ({method} feature {i}): {e}")
            importances[i] = 0.0

    return dict(zip(names, importances.tolist()))

def compute_permutation_importance(model, X_val, y_val, feature_names, **kwargs):
    """
    ✅ 기존 permutation importance 함수 → compute_feature_importance와 동일 방식으로 통일
    """
    return compute_feature_importance(model, X_val, y_val, feature_names, method="permutation", **kwargs)

def save_feature_importance(importances, symbol, strategy, model_type, method="baseline"):
    suffix = f"_importance_{method}"
    fname_json = f"{symbol}_{strategy}_{model_type}{suffix}.json"
    fname_csv = f"{symbol}_{strategy}_{model_type}{suffix}.csv"
    path_json = os.path.join(IMPORTANCE_DIR, fname_json)
    path_csv = os.path.join(IMPORTANCE_DIR, fname_csv)

    importances = {str(k): float(v) for k, v in (importances or {}).items()}

    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(importances, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(importances.items(), columns=["feature", "importance"]).sort_values(by="importance", ascending=False)
    df.to_csv(path_csv, index=False, encoding="utf-8-sig")

    print(f"✅ 중요도 저장 완료: {path_json}, {path_csv}")

def drop_low_importance_features(
    df: pd.DataFrame,
    importances: dict,
    threshold: float = 0.05,
    input_size: int = None,
    min_features: int = 5
) -> pd.DataFrame:
    """
    ✅ feature importance 기반 low-importance feature drop 함수
    - threshold 이하 feature 제거
    - 최소 min_features 개수 유지 (부족 시 pad 컬럼 추가)
    """
    importances = importances or {}
    drop_cols = [col for col, imp in importances.items() if (imp is not None and float(imp) < threshold)]
    remaining_cols = [c for c in df.columns if c not in drop_cols and c not in ["timestamp", "strategy"]]

    # ✅ 최소 개수 유지
    if len(remaining_cols) < min_features:
        for i in range(len(remaining_cols), min_features):
            pad_col = f"pad_{i}"
            if pad_col not in df.columns:
                df[pad_col] = 0.0
            remaining_cols.append(pad_col)

    # ✅ 모든 컬럼 제거 방지
    if not remaining_cols:
        print("[경고] 모든 feature가 제거되었음. pad_0 컬럼 추가")
        df["pad_0"] = 0.0
        remaining_cols = ["pad_0"]

    print(f"🧹 제거된 feature 수: {len(drop_cols)} → {drop_cols}")
    cols_out = remaining_cols + [c for c in ["timestamp", "strategy"] if c in df.columns]
    return df[cols_out]

def get_top_features(importances: dict, top_n: int = 10) -> pd.DataFrame:
    if not importances:
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.DataFrame(importances.items(), columns=["feature", "importance"])
    return df.sort_values(by="importance", ascending=False).head(top_n)
