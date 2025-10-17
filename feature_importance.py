# === feature_importance.py (디스크 부족 내성, 지연 디렉토리 생성, 안전 저장) ===
import os
import json
import time
import errno
import torch
import numpy as np
import pandas as pd

PERSIST_DIR = "/persistent"
DEFAULT_IMPORTANCE_DIR = os.path.join(PERSIST_DIR, "importances")
FALLBACK_IMPORTANCE_DIR = os.environ.get("TMPDIR", "/tmp") + "/importances"
DISABLE_IMPORTANCE_SAVE = os.environ.get("DISABLE_IMPORTANCE_SAVE", "0") == "1"

# import 시 디렉토리 생성 금지 → 저장 시점에 안전하게 처리
_cached_dir = None
_cached_disabled = False

def _ensure_dir(path: str) -> str | None:
    """디렉토리 생성. ENOSPC면 None 반환. 그 외 오류는 재시도 후 포기."""
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except OSError as e:
        if e.errno == errno.ENOSPC:
            return None
        # 읽기 전용 등 기타 오류
        return None

def _get_importance_dir() -> str | None:
    """우선순위: /persistent → /tmp. 둘 다 실패 시 None."""
    global _cached_dir, _cached_disabled
    if _cached_disabled or DISABLE_IMPORTANCE_SAVE:
        return None
    if _cached_dir:
        return _cached_dir
    # 1) /persistent
    d1 = _ensure_dir(DEFAULT_IMPORTANCE_DIR)
    if d1:
        _cached_dir = d1
        return d1
    # 2) /tmp
    d2 = _ensure_dir(FALLBACK_IMPORTANCE_DIR)
    if d2:
        _cached_dir = d2
        print(f"[feature_importance] 저장 경로를 임시로 전환: {d2}")
        return d2
    # 3) 비활성화
    _cached_disabled = True
    print("[feature_importance] 경고: 저장 비활성화(디스크 여유 없음)")
    return None

def _ensure_feature_names(feature_names, n_features):
    if isinstance(feature_names, (list, tuple)) and len(feature_names) == n_features:
        return list(feature_names)
    return [f"f_{i}" for i in range(n_features)]

@torch.no_grad()
def compute_feature_importance(
    model,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    feature_names,
    method: str = "baseline",
    *,
    max_seconds: float = 30.0,
    print_every: int = 20
):
    """
    permutation 기반 중요도. 시간 예산 초과 시 조기 종료.
    X_val: (N, T, F)
    """
    model.eval()
    start = time.time()

    try:
        logits = model(X_val)
        y_val = y_val.view(-1).long()
        baseline_loss = torch.nn.CrossEntropyLoss()(logits, y_val).item()
    except Exception as e:
        print(f"[feature_importance] 모델 예측 실패({method}): {e}")
        n_feat = int(X_val.shape[2])
        names = _ensure_feature_names(feature_names, n_feat)
        return dict(zip(names, [0.0] * n_feat))

    n_feat = int(X_val.shape[2])
    names = _ensure_feature_names(feature_names, n_feat)
    importances = np.zeros(n_feat, dtype=float)

    for i in range(n_feat):
        if (time.time() - start) > float(max_seconds):
            print(f"[feature_importance] 시간예산 초과 → {i}/{n_feat}에서 중단(잔여=0)")
            break
        try:
            X_perm = X_val.clone()
            perm_idx = torch.randperm(X_val.shape[0], device=X_val.device)
            X_perm[:, :, i] = X_perm[perm_idx, :, i]
            logits_perm = model(X_perm)
            loss = torch.nn.CrossEntropyLoss()(logits_perm, y_val).item()
            importances[i] = float(loss - baseline_loss)
            if print_every and (i % max(1, int(print_every)) == 0):
                print(f"[imp] {i+1}/{n_feat} Δ={importances[i]:.6f}")
        except Exception as e:
            print(f"[feature_importance] 중요도 계산 실패({method}, idx={i}): {e}")
            importances[i] = 0.0

    return dict(zip(names, importances.tolist()))

def compute_permutation_importance(model, X_val, y_val, feature_names, **kwargs):
    return compute_feature_importance(model, X_val, y_val, feature_names, method="permutation", **kwargs)

def _safe_write_json(path: str, obj) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return True
    except OSError as e:
        if getattr(e, "errno", None) == errno.ENOSPC:
            print(f"[feature_importance] ENOSPC: JSON 저장 실패 → {path}")
            return False
        print(f"[feature_importance] JSON 저장 오류: {e}")
        return False

def _safe_write_csv(path: str, df: pd.DataFrame) -> bool:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return True
    except OSError as e:
        if getattr(e, "errno", None) == errno.ENOSPC:
            print(f"[feature_importance] ENOSPC: CSV 저장 실패 → {path}")
            return False
        print(f"[feature_importance] CSV 저장 오류: {e}")
        return False

def save_feature_importance(importances, symbol, strategy, model_type, method="baseline"):
    """
    디스크 부족 시 /tmp로 자동 폴백. 그래도 실패하면 저장 생략.
    """
    if DISABLE_IMPORTANCE_SAVE:
        print("[feature_importance] 저장 비활성화(환경변수)")
        return False

    out_dir = _get_importance_dir()
    if not out_dir:
        # 저장 불가. 조용히 패스하되 호출부는 계속 진행 가능.
        return False

    suffix = f"_importance_{method}"
    fname_base = f"{symbol}_{strategy}_{model_type}{suffix}"
    path_json = os.path.join(out_dir, fname_base + ".json")
    path_csv = os.path.join(out_dir, fname_base + ".csv")

    imp = {str(k): float(v) for k, v in (importances or {}).items()}
    ok_json = _safe_write_json(path_json, imp)

    df = pd.DataFrame(imp.items(), columns=["feature", "importance"]).sort_values(by="importance", ascending=False)
    ok_csv = _safe_write_csv(path_csv, df)

    if ok_json and ok_csv:
        print(f"[feature_importance] 저장 완료: {path_json}, {path_csv}")
        return True

    # json 또는 csv 중 1개라도 실패하면 한 번 더 /tmp로 시도
    if out_dir != FALLBACK_IMPORTANCE_DIR:
        fb_dir = _ensure_dir(FALLBACK_IMPORTANCE_DIR)
        if fb_dir:
            pj = os.path.join(fb_dir, fname_base + ".json")
            pc = os.path.join(fb_dir, fname_base + ".csv")
            ok_json2 = ok_json or _safe_write_json(pj, imp)
            ok_csv2 = ok_csv or _safe_write_csv(pc, df)
            if ok_json2 and ok_csv2:
                print(f"[feature_importance] 폴백 저장 완료: {pj}, {pc}")
                return True

    print("[feature_importance] 경고: 중요도 저장 실패(디스크 여유 없음)")
    return False

def drop_low_importance_features(
    df: pd.DataFrame,
    importances: dict,
    threshold: float = 0.05,
    input_size: int = None,
    min_features: int = 5
) -> pd.DataFrame:
    importances = importances or {}
    drop_cols = [col for col, imp in importances.items() if (imp is not None and float(imp) < threshold)]
    remaining_cols = [c for c in df.columns if c not in drop_cols and c not in ["timestamp", "strategy"]]

    if len(remaining_cols) < min_features:
        for i in range(len(remaining_cols), min_features):
            pad_col = f"pad_{i}"
            if pad_col not in df.columns:
                df[pad_col] = 0.0
            remaining_cols.append(pad_col)

    if not remaining_cols:
        print("[feature_importance] 경고: 모든 feature 제거됨 → pad_0 추가")
        df["pad_0"] = 0.0
        remaining_cols = ["pad_0"]

    print(f"[feature_importance] 제거된 feature 수: {len(drop_cols)} → {drop_cols}")
    cols_out = remaining_cols + [c for c in ["timestamp", "strategy"] if c in df.columns]
    return df[cols_out]

def get_top_features(importances: dict, top_n: int = 10) -> pd.DataFrame:
    if not importances:
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.DataFrame(importances.items(), columns=["feature", "importance"])
    return df.sort_values(by="importance", ascending=False).head(top_n)
