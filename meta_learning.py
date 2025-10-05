# meta_learning.py
# ------------------------------------------------------------
# 단일 진입점 메타러너 (NaN/None 방어, 과도 abstain 방지, EVO/스태킹/룰 폴백)
# predict.py의 포지션 힌트/최소 기대수익 필터와 정합
# ------------------------------------------------------------

from __future__ import annotations
import os
import math
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

# (선택) 토치 의존은 존재할 때만 사용
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False
    torch = None
    nn = object  # type: ignore
    optim = object  # type: ignore

# (선택) 1단계 evo_meta_learner 연동
_EVO_OK = False
try:
    from evo_meta_learner import (
        predict_evo_meta,               # (X_new, input_size, probs_stack=None) -> int or None
        aggregate_probs_for_meta        # (W,C)->(C,)
    )
    _EVO_OK = True
except Exception:
    _EVO_OK = False

# ===== (환경설정) 성공률 기본값: 성공 이력 없을 때 prior =====
META_BASE_SUCCESS = float(os.getenv("META_BASE_SUCCESS", "0.55"))

# 기대수익률 임계(절댓값). 미만은 0 처리 → 선택 영향 제거
_RET_TH = float(os.getenv("META_ER_THRESHOLD", "0.01"))

# predict.py와 정합: 기본 최소 기대수익(절댓값) — 필요 시 인자로 덮어씀
META_MIN_RETURN = float(os.getenv("META_MIN_RETURN", "0.01"))

# EVO-메타 합성 규칙 환경변수
EVO_META_AGG = os.getenv("EVO_META_AGG", "mean_var").lower()   # mean | varpen | mean_var
EVO_META_VAR_GAMMA = float(os.getenv("EVO_META_VAR_GAMMA", "1.0"))

# ===== 폭·신뢰도 보정 관련 환경변수 =====
CLAMP_MAX_WIDTH = float(os.getenv("CLAMP_MAX_WIDTH", "0.10"))
META_CI_Z = float(os.getenv("META_CI_Z", "1.64"))  # ~90% CI
META_MIN_N = int(os.getenv("META_MIN_N", "30"))

# ======================= (A) MAML (유지) =======================
if _TORCH_OK:
    class MAML:
        def __init__(self, model, inner_lr=0.01, outer_lr=0.001, inner_steps=1):
            self.model = model
            self.inner_lr = inner_lr
            self.outer_lr = outer_lr
            self.inner_steps = inner_steps
            self.optimizer = optim.Adam(self.model.parameters(), lr=outer_lr)
            self.loss_fn = nn.CrossEntropyLoss()

        def adapt(self, X, y):
            adapted_params = {name: p for name, p in self.model.named_parameters()}
            for _ in range(self.inner_steps):
                logits = (self.model.forward(X, params=adapted_params)
                          if hasattr(self.model, 'forward') else self.model(X))
                loss = self.loss_fn(logits, y)
                grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
                adapted_params = {name: p - self.inner_lr * g
                                  for (name, p), g in zip(adapted_params.items(), grads)}
            return adapted_params

        def meta_update(self, tasks):
            meta_loss = 0.0
            for X_tr, y_tr, X_va, y_va in tasks:
                adapted = self.adapt(X_tr, y_tr)
                logits = (self.model.forward(X_va, params=adapted)
                          if hasattr(self.model, 'forward') else self.model(X_va))
                meta_loss = meta_loss + self.loss_fn(logits, y_va)
            if len(tasks) == 0:
                return 0.0
            meta_loss = meta_loss / float(len(tasks))
            self.optimizer.zero_grad()
            meta_loss.backward()
            self.optimizer.step()
            return float(meta_loss.item())

def maml_train_entry(model, train_loader, val_loader, inner_lr=0.01, outer_lr=0.001, inner_steps=1):
    if not _TORCH_OK:
        print("[MAML skip] torch 미존재 → meta update 생략")
        return None
    try:
        maml = MAML(model, inner_lr=inner_lr, outer_lr=outer_lr, inner_steps=inner_steps)
        tasks = []
        for (X_tr, y_tr), (X_va, y_va) in zip(train_loader, val_loader):
            tasks.append((X_tr, y_tr, X_va, y_va))
        if not tasks:
            print("[MAML skip] 유효한 meta task 없음 → meta update 생략")
            return None
        loss = maml.meta_update(tasks)
        print(f"[✅ MAML meta-update 완료] task={len(tasks)}, loss={loss:.4f}")
        return loss
    except Exception as e:
        print(f"[❌ MAML 예외 발생] {e}")
        return None


# ================= (B) 스태킹형 메타러너(Scikit) =================
import pickle

META_MODEL_PATH = "/persistent/models/meta_learner.pkl"

def train_meta_learner(model_outputs_list, true_labels):
    try:
        from sklearn.linear_model import LogisticRegression
        X = [np.array(mo).flatten() for mo in model_outputs_list]
        y = np.array(true_labels)
        clf = LogisticRegression(max_iter=500)
        clf.fit(X, y)
        os.makedirs(os.path.dirname(META_MODEL_PATH), exist_ok=True)
        with open(META_MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)
        print(f"[✅ meta learner 학습 완료 및 저장] {META_MODEL_PATH}")
        return clf
    except Exception as e:
        print(f"[⚠️ meta learner 학습 실패: {e}]")
        return None

def load_meta_learner():
    try:
        if os.path.exists(META_MODEL_PATH):
            with open(META_MODEL_PATH, "rb") as f:
                clf = pickle.load(f)
            print("[✅ meta learner 로드 완료]")
            return clf
    except Exception as e:
        print(f"[⚠️ meta learner 로드 실패: {e}]")
    print("[⚠️ meta learner 파일 없음 → None 반환]")
    return None


# ========== (C) 안전 로그 유틸(있으면 사용, 없으면 프린트) ==========
def _safe_log_prediction(**kwargs):
    try:
        from logger import log_prediction  # 선택적 의존
        # 숫자 보장을 위해 한번 더 nan→0 처리
        for k in ("rate", "return_value", "entry_price", "target_price"):
            if k in kwargs:
                kwargs[k] = float(np.nan_to_num(kwargs[k], nan=0.0, posinf=0.0, neginf=0.0))
        log_prediction(**kwargs)
    except Exception:
        info = {k: kwargs.get(k) for k in ["symbol", "strategy", "model",
                                           "predicted_class", "note", "rate", "reason", "source"]}
        print(f"[META-LOG Fallback] {info}")


# ================= (D) 안전 정규화/집계 유틸 =================

# ▼ 추가: 범용 나노가드(스칼라/벡터 모두 숫자 강제) -----------------
def _nan_guard(x: Any, *, fill: float = 0.0):
    try:
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=np.float64)
            arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
            return arr
        # 스칼라
        return float(np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill))
    except Exception:
        return fill
# ----------------------------------------------------------------

def _normalize_safe(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    v[~np.isfinite(v)] = 0.0
    v[v < 0] = 0.0
    s = float(v.sum())
    if s <= 0 or not np.isfinite(s):
        # 완전 망가졌을 때 균등분포
        n = max(1, len(v))
        return np.ones(n, dtype=np.float64) / n
    return v / s

def _to_probs(x: Any, C_expected: Optional[int] = None) -> Optional[np.ndarray]:
    """
    입력을 확률 벡터로 강제 변환.
    - 허용: list/ndarray, dict{class:prob}, torch.Tensor
    - NaN/Inf/음수/합계0 → 자동 복구
    - 길이 불일치 시 None 반환(호출부에서 스킵)
    """
    try:
        if x is None:
            return None
        if isinstance(x, dict):
            if not x:
                return None
            # dict는 key를 0..C-1로 맞춰야 함
            C = max(int(k) for k in x.keys()) + 1
            if C_expected is not None and C_expected != C:
                return None
            vec = np.zeros(C, dtype=np.float64)
            for k, v in x.items():
                ki = int(k)
                try:
                    vec[ki] = float(v)
                except Exception:
                    vec[ki] = 0.0
            return _normalize_safe(vec)
        arr = np.array(x, dtype=np.float64).reshape(-1)
        if C_expected is not None and arr.size != C_expected:
            return None
        return _normalize_safe(arr)
    except Exception:
        return None

def _entropy(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

def _aggregate_base_outputs(
    groups_outputs: List[Dict],
    class_success: Optional[Dict[int, float]] = None,
    mode: str = "avg"
) -> Tuple[np.ndarray, Dict]:
    """
    groups_outputs: [{"probs": vector|dict|tensor, "group_id": int, ...}, ...]
    class_success:  {cls: success_rate in [0,1]}
    mode: "avg" | "weighted" | "maxvote"
    """
    if not groups_outputs:
        raise ValueError("groups_outputs 비어있음")

    # 첫 유효 항목에서 클래스 수 추정
    C_guess = None
    for g in groups_outputs:
        p = g.get("probs")
        arr = _to_probs(p, None)
        if arr is not None:
            C_guess = arr.size
            break
    if C_guess is None:
        # 정말 모두 망가지면 2클래스 가정하여 균등
        C_guess = 2

    probs_mat = []
    for g in groups_outputs:
        arr = _to_probs(g.get("probs"), C_guess)
        if arr is None:
            continue
        probs_mat.append(arr)
    if not probs_mat:
        # 모든 항목이 무효 → 균등으로 시작
        agg = np.ones(C_guess, dtype=np.float64) / C_guess
        detail = {"entropy": _entropy(agg), "top1": int(np.argmax(agg)),
                  "top1_prob": float(agg.max()), "margin": 0.0,
                  "probs_stack_shape": [0, C_guess], "fallback": "all_invalid_uniform"}
        return agg.astype(np.float32), detail

    probs_mat = np.stack(probs_mat, axis=0)  # (N, C)
    C = probs_mat.shape[1]

    detail: Dict = {}
    if mode == "avg":
        agg = probs_mat.mean(axis=0)

    elif mode == "weighted":
        if not class_success:
            agg = probs_mat.mean(axis=0)
        else:
            w = np.array([class_success.get(c, META_BASE_SUCCESS) for c in range(C)], dtype=np.float64)
            w = (w - w.min()) / (w.max() - w.min() + 1e-9)
            w = 0.5 + 0.5 * w
            agg = (probs_mat * w).mean(axis=0)
            detail["class_weights"] = w.tolist()

    elif mode == "maxvote":
        votes = np.bincount(np.argmax(probs_mat, axis=1), minlength=C)
        agg = votes.astype(np.float64) / max(1, votes.sum())
        detail["votes"] = votes.tolist()

    else:
        agg = probs_mat.mean(axis=0)  # 알 수 없는 모드 → 평균 폴백
        detail["fallback"] = f"unknown_agg:{mode}"

    agg = _normalize_safe(agg)
    detail["entropy"] = _entropy(agg)
    detail["top1"] = int(np.argmax(agg))
    detail["top1_prob"] = float(agg[detail["top1"]])
    top2 = float(np.partition(agg, -2)[-2]) if C >= 2 else 0.0
    detail["margin"] = float(detail["top1_prob"] - top2)
    detail["probs_stack_shape"] = list(probs_mat.shape)
    return agg.astype(np.float32), detail


# ======= 폭·신뢰도 보정 유틸 =======
def _ret_gain(er: float) -> float:
    er_abs = abs(float(er))
    return max(0.0, er_abs - _RET_TH)

def _extract_counts(meta_info: Dict) -> Dict[int, int]:
    if not isinstance(meta_info, dict):
        return {}
    cand = {}
    for k in ["counts", "trial_counts", "samples"]:
        v = meta_info.get(k)
        if isinstance(v, dict):
            cand = {int(c): int(vv) for c, vv in v.items() if isinstance(vv, (int, float))}
            if cand:
                return cand
    v = meta_info.get("success_total")
    if isinstance(v, dict):
        out = {}
        for c, pair in v.items():
            try:
                s, n = pair
                out[int(c)] = int(n)
            except Exception:
                pass
        if out:
            return out
    return {}

def _adjust_success_rates_with_ci(
    sr_dict: Dict[int, float],
    counts: Dict[int, int],
    z: float = META_CI_Z,
    n0: int = META_MIN_N,
    prior: float = META_BASE_SUCCESS
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for c, sr in sr_dict.items():
        try:
            sr = float(sr)
        except Exception:
            continue
        sr = max(0.0, min(1.0, sr))
        n = int(counts.get(c, 0))
        sr_blend = (n / (n + n0)) * sr + (n0 / (n + n0)) * prior if n >= 0 else sr
        if n > 0:
            se = math.sqrt(max(1e-12, sr_blend * (1.0 - sr_blend) / n))
            lo = max(0.0, sr_blend - z * se)
            out[int(c)] = float(lo)
        else:
            out[int(c)] = float(max(0.0, prior - z * math.sqrt(prior * (1 - prior) / (n0 + 1e-9))))
    return out

def _width_scaled_er(
    er_dict: Dict[int, float],
    class_ranges: Optional[List[Tuple[float, float]]] = None,
    max_width: float = CLAMP_MAX_WIDTH
) -> Dict[int, float]:
    if not class_ranges:
        return {int(c): float(er) for c, er in er_dict.items()}
    out: Dict[int, float] = {}
    for c, er in er_dict.items():
        try:
            lo, hi = class_ranges[int(c)]
            width = float(hi) - float(lo)
        except Exception:
            width = None
        if width is None or width <= 0:
            out[int(c)] = float(er)
            continue
        scale = min(1.0, max(1e-9, float(max_width) / abs(width)))
        out[int(c)] = float(er) * scale
    return out


# ======= 포지션/최소 기대수익 필터 =======
def _position_from_range(lo: float, hi: float) -> str:
    try:
        lo = float(lo); hi = float(hi)
        if hi <= 0 and lo < 0: return "short"
        if lo >= 0 and hi > 0: return "long"
        return "neutral"
    except Exception:
        return "neutral"

def _mask_by_hint_and_minret(
    scores: np.ndarray,
    class_ranges: Optional[List[Tuple[float, float]]],
    *,
    allow_long: bool,
    allow_short: bool,
    min_return_thr: float
) -> Tuple[np.ndarray, Dict]:
    C = len(scores)
    reasons: Dict[int, str] = {}
    if not class_ranges or len(class_ranges) < C:
        return scores, reasons

    out = scores.astype(np.float64).copy()
    for c in range(C):
        try:
            lo, hi = class_ranges[c]
            mid = 0.5 * (float(lo) + float(hi))
            pos = _position_from_range(lo, hi)
            if abs(mid) < float(min_return_thr):
                out[c] = 0.0
                reasons[c] = f"minret({abs(mid):.4f}<{min_return_thr:.4f})"
                continue
            if pos == "long" and not allow_long:
                out[c] = 0.0
                reasons[c] = "hint_block_long"
                continue
            if pos == "short" and not allow_short:
                out[c] = 0.0
                reasons[c] = "hint_block_short"
                continue
        except Exception:
            pass
    return out, reasons


# ========== (E) 단독 유틸: 성공률/수익률 고려 최종 클래스 산출 ==========
def get_meta_prediction(model_outputs_list, feature_tensor=None, meta_info=None):
    """
    기본 유틸: 성공률/수익률 고려 스코어로 최종 클래스 산출
    (과도 abstain 방지 — 항상 유효 클래스 반환)
    """
    if not model_outputs_list:
        raise ValueError("❌ get_meta_prediction: 모델 출력 없음")

    # 첫 유효 확률에서 C 추정
    C_guess = None
    for m in model_outputs_list:
        probs = m["probs"] if isinstance(m, dict) else m
        arr = _to_probs(probs, None)
        if arr is not None:
            C_guess = arr.size
            break
    if C_guess is None:
        C_guess = 2

    softmax_list = []
    for m in model_outputs_list:
        probs = m["probs"] if isinstance(m, dict) else m
        arr = _to_probs(probs, C_guess)
        if arr is not None:
            softmax_list.append(arr)
    if not softmax_list:
        avg_softmax = np.ones(C_guess, dtype=np.float64) / C_guess
    else:
        avg_softmax = _normalize_safe(np.mean(softmax_list, axis=0))

    meta_info = meta_info or {}
    success_rate_dict = dict(meta_info.get("success_rate", {}))
    expected_return_dict = dict(meta_info.get("expected_return", {}))
    class_ranges = meta_info.get("class_ranges", None)

    # 성공률 신뢰구간 보정
    counts = _extract_counts(meta_info)
    if success_rate_dict:
        success_rate_dict = _adjust_success_rates_with_ci(success_rate_dict, counts)

    # 폭 기반 기대수익 영향도 축소
    if expected_return_dict:
        expected_return_dict = _width_scaled_er(expected_return_dict, class_ranges, CLAMP_MAX_WIDTH)

    scores = np.zeros_like(avg_softmax, dtype=np.float64)
    all_below = True
    for c in range(len(avg_softmax)):
        sr = float(success_rate_dict.get(c, META_BASE_SUCCESS))
        er = float(expected_return_dict.get(c, 0.0))
        g  = _ret_gain(er)
        if g > 0:
            all_below = False
        scores[c] = float(avg_softmax[c]) * sr * g

    mode = "성공률 기반 메타(CI/폭보정)" if (success_rate_dict or expected_return_dict) else "기본 메타 (성공률/ER 無)"
    if all_below:
        scores = avg_softmax.copy()
        mode += " / all<TH→prob선택"

    # ▼ 숫자 보장
    scores = _nan_guard(scores)
    final_pred_class = int(np.argmax(scores))
    print(f"[META] {mode} → 최종 클래스 {final_pred_class} / 점수={np.round(scores, 4)}")
    return final_pred_class


# ================= (F) EVO-메타 연동 보조 =================
def _build_evo_meta_vector(agg_probs: np.ndarray, expected_return: Dict[int, float]) -> np.ndarray:
    C = int(len(agg_probs))
    vec = []
    for c in range(C):
        p = float(agg_probs[c])
        er = float(expected_return.get(c, 0.0))
        vec.extend([p, er, 0.0])  # hit=0
    return np.array(vec, dtype=np.float32)

def _maybe_evo_decide(
    groups_outputs: List[Dict],
    agg_probs: np.ndarray,
    expected_return: Dict[int, float],
) -> Optional[int]:
    if not _EVO_OK:
        return None
    try:
        probs_stack = np.stack(
            [_normalize_safe(_to_probs(g["probs"], len(agg_probs))) for g in groups_outputs if _to_probs(g.get("probs"), len(agg_probs)) is not None],
            axis=0
        )
        try:
            _ = aggregate_probs_for_meta(probs_stack, mode=EVO_META_AGG, gamma=EVO_META_VAR_GAMMA)
        except Exception:
            pass
        X_new = _build_evo_meta_vector(agg_probs, expected_return)
        pred = predict_evo_meta(X_new, input_size=int(X_new.shape[0]), probs_stack=probs_stack)
        if isinstance(pred, (int, np.integer)):
            return int(pred)
    except Exception as e:
        print(f"[⚠️ EVO 메타 예측 실패] {e}")
    return None


# ========== (G) 단일 진입점: meta_predict(...) ==========
def meta_predict(
    symbol: str,
    horizon: str,
    groups_outputs: List[Dict],
    features: Optional["torch.Tensor"] = None,
    meta_state: Optional[Dict] = None,
    *,
    agg_mode: str = "avg",
    use_stacking: bool = True,
    use_evo_meta: bool = True,
    log: bool = True,
    source: str = "meta",
    # ▼ predict.py와 정합되는 필터 인자 ▼
    position_hint: Optional[Dict[str, bool]] = None,  # {"allow_long": bool, "allow_short": bool}
    min_return_thr: Optional[float] = None
) -> Dict:
    """
    단일 진입점
    - 베이스 모델 출력 집계 + EVO-메타/스태킹/룰기반 폴백
    - 성공률/실패율/수익률을 모두 고려
    - 포지션 힌트/최소 기대수익 마스킹을 확률/스코어 레벨에 적용
    - 과도 abstain 방지: 항상 유효 클래스 반환(보류 여부는 predict.py 정책)
    """
    meta_state = meta_state or {}
    class_success_raw = dict(meta_state.get("success_rate", {}))
    expected_return_raw = dict(meta_state.get("expected_return", {}))
    class_ranges = meta_state.get("class_ranges", None)

    allow_long = bool((position_hint or {}).get("allow_long", True))
    allow_short = bool((position_hint or {}).get("allow_short", True))
    min_thr = float(min_return_thr if min_return_thr is not None else max(META_MIN_RETURN, _RET_TH))

    # (1) 집계 (NaN/형태 오류 자동 스킵)
    agg_probs, detail = _aggregate_base_outputs(groups_outputs, class_success_raw, mode=agg_mode)
    used_mode = agg_mode
    final_class = int(np.argmax(agg_probs))

    # 성공률 CI 보정 + 폭 보정된 기대수익
    counts = _extract_counts(meta_state)
    class_success_ci = _adjust_success_rates_with_ci(class_success_raw, counts) if class_success_raw else {}
    expected_return_scaled = _width_scaled_er(expected_return_raw, class_ranges, CLAMP_MAX_WIDTH) if expected_return_raw else {}

    # (1.5) 확률에 힌트/최소 기대수익 마스크 1차 적용
    probs_masked, mask_reasons_p = _mask_by_hint_and_minret(
        agg_probs, class_ranges,
        allow_long=allow_long, allow_short=allow_short, min_return_thr=min_thr
    )
    if probs_masked.sum() > 0:
        agg_probs = _normalize_safe(probs_masked)
        detail.setdefault("filters", {})["prob_mask"] = mask_reasons_p
    else:
        # 전부 0이 되면 과도 마스킹 → 마스크 무시 폴백
        detail.setdefault("filters", {})["prob_mask"] = {"_fallback": "all_zero → ignore_mask"}
        agg_probs = _normalize_safe(agg_probs)

    # (2) EVO 메타 (가능하면 최우선)
    evo_choice: Optional[int] = None
    if use_evo_meta:
        evo_choice = _maybe_evo_decide(groups_outputs, agg_probs, expected_return_scaled)
        if isinstance(evo_choice, int):
            used_mode = "evo_meta"
            final_class = int(evo_choice)

    # (3) 스태킹 (EVO 실패 시)
    if use_stacking and used_mode != "evo_meta":
        try:
            clf = load_meta_learner()
            if clf is not None:
                X_stack = np.concatenate(
                    [_normalize_safe(_to_probs(g["probs"], len(agg_probs))) for g in groups_outputs
                     if _to_probs(g.get("probs"), len(agg_probs)) is not None],
                    axis=0
                ).reshape(1, -1)
                stacked_pred = clf.predict(X_stack)[0]
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(X_stack)[0]
                    final_class = int(stacked_pred)
                    used_mode = "stacking"
                    if len(proba) == len(agg_probs):
                        agg_probs = _normalize_safe(proba.astype(np.float64)).astype(np.float32)
                else:
                    final_class = int(stacked_pred)
                    used_mode = "stacking(base-probs)"
        except Exception as e:
            print(f"[⚠️ stacking 예측 실패 → 집계 폴백] {e}")

    # (4) 신뢰도 점검 + 룰기반 보정 (성공률/ER 가중치 스코어)
    top1 = int(np.argmax(agg_probs))
    top1p = float(agg_probs[top1])
    margin = float(top1p - float(np.partition(agg_probs, -2)[-2]) if len(agg_probs) >= 2 else top1p)
    ent = _entropy(agg_probs)

    scores = agg_probs.astype(np.float64).copy()
    C = len(scores)
    all_er_below = True
    for c in range(C):
        sr = float(class_success_ci.get(c, META_BASE_SUCCESS))
        sr = max(0.0, min(1.0, sr))
        fr = 1.0 - sr
        er = float(expected_return_scaled.get(c, 0.0))
        g = _ret_gain(er)
        if g > 0:
            all_er_below = False
        scores[c] = scores[c] * sr * g * (1.0 - 0.3 * fr)

    # (4.1) 스코어에도 힌트/최소 기대수익 마스크 2차 적용
    scores_masked, mask_reasons_s = _mask_by_hint_and_minret(
        scores, class_ranges,
        allow_long=allow_long, allow_short=allow_short, min_return_thr=min_thr
    )
    if scores_masked.sum() > 0:
        scores = scores_masked
        detail.setdefault("filters", {})["score_mask"] = mask_reasons_s
    else:
        # 전부 0이면 과도 마스킹 → 확률 기반으로 폴백
        scores = agg_probs.astype(np.float64).copy()
        detail.setdefault("filters", {})["score_mask"] = {"_fallback": "all_zero → use_probs"}

    low_conf = (margin < 0.05) or (ent > math.log(max(2, len(agg_probs))) * 0.8)

    # 기대수익 전부 임계 미만이거나 스코어 합 0이면 확률로 폴백(마스크 적용 결과가 0이면 균등 분포)
    if all_er_below or scores.sum() == 0:
        tmp, _ = _mask_by_hint_and_minret(
            agg_probs.copy(), class_ranges,
            allow_long=allow_long, allow_short=allow_short, min_return_thr=min_thr
        )
        scores = tmp if tmp.sum() > 0 else (np.ones_like(agg_probs, dtype=np.float64) / len(agg_probs))
        low_conf = True

    # ▼ 숫자 보장
    scores = _nan_guard(scores)
    agg_probs = _nan_guard(agg_probs)

    rule_choice = int(np.argmax(scores))
    detail["final_scores"] = np.round(scores, 4).tolist()
    detail["success_rate_ci"] = {int(k): float(v) for k, v in class_success_ci.items()}
    detail["expected_return_scaled"] = {int(k): float(v) for k, v in expected_return_scaled.items()}
    detail["ci_z"] = float(META_CI_Z)
    detail["clamp_max_width"] = float(CLAMP_MAX_WIDTH)
    detail["min_return_thr"] = float(min_thr)
    detail["hint_allow_long"] = bool(allow_long)
    detail["hint_allow_short"] = bool(allow_short)

    # 최종 선택: EVO > 스태킹 > 집계, 단 낮은 신뢰도면 rule_choice
    if low_conf:
        used_mode = "rule_fallback"
        final_class = rule_choice
    else:
        if rule_choice != final_class and scores[rule_choice] >= scores[final_class] * 1.1:
            used_mode = "rule_bias"
            final_class = rule_choice

    # 방어: 인덱스 범위 보장
    if not (0 <= int(final_class) < len(agg_probs)):
        final_class = int(np.argmax(agg_probs))
        used_mode += "+idx_guard"

    result = {
        "class": int(final_class),
        "probs": np.asarray(agg_probs, dtype=np.float32).tolist(),
        "confidence": float(np.nan_to_num(max(agg_probs), nan=0.0, posinf=0.0, neginf=0.0)),
        "margin": float(np.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)),
        "entropy": float(np.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0)),
        "mode": used_mode,
        "detail": detail,
    }

    # (5) 로깅
    if log:
        er_cho = 0.0
        try:
            if class_ranges and 0 <= result["class"] < len(class_ranges):
                lo, hi = class_ranges[result["class"]]
                er_cho = 0.5 * (float(lo) + float(hi))
        except Exception:
            pass
        sr_cho = class_success_ci.get(result["class"], None)
        note = (f"meta:{used_mode} top1={result['class']} p={result['confidence']:.3f} "
                f"margin={result['margin']:.3f} ERmid={er_cho:.4f} "
                f"SR={('-' if sr_cho is None else f'{float(sr_cho):.2f}')} "
                f"TH={_RET_TH:.2%} EVO_AGG={EVO_META_AGG} γ={EVO_META_VAR_GAMMA} "
                f"CIz={META_CI_Z:.2f} Wmax={CLAMP_MAX_WIDTH:.3f} "
                f"allowL={allow_long} allowS={allow_short} minER={min_thr:.4f}")
        reason = f"entropy={result['entropy']:.3f}"
        _safe_log_prediction(
            symbol=symbol,
            strategy=horizon,
            direction="메타예측",
            entry_price=0,
            target_price=0,
            model="meta",
            model_name=used_mode,
            predicted_class=result["class"],
            label=result["class"],
            note=note,
            success=True,
            reason=reason,
            rate=0.0,
            return_value=0.0,
            volatility=False,
            source=source,
            group_id=0
        )

    print(f"[META] mode={used_mode} class={result['class']} "
          f"conf={result['confidence']:.3f} margin={result['margin']:.3f} "
          f"entropy={result['entropy']:.3f}")
    return result
