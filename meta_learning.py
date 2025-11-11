# meta_learning.py (YOPO v1.7 — 하이브리드 진화판 연동 / 심볼-전략 일치 가드 강화 / 예측기와 동일 구간보정)
# ------------------------------------------------------------------
# 변경 요약
# 1) predict.py 가 뿌려주는 필드(hybrid_probs, adjusted_probs, filtered_probs, success_score 등)를
#    그대로 우선 사용하도록 집계부를 확장함. → "확률 + 유사도"가 여기서도 그대로 이어짐.
# 2) class_ranges 단위가 퍼센트로 들어오는 경우를 predict.py 와 같은 방식으로 자동 보정(_sanitize_range)
#    해서 "0.04 = 4%" / "4 = 4%" 혼선을 제거함.
# 3) meta_state.class_ranges 를 보정한 뒤 힌트/최소수익률 필터에 사용하도록 일원화.
# 4) symbol / strategy 불일치 후보 제거는 유지하되, 제거된 수와 이유를 log note 에 남김.
# 5) log_prediction 의 model_name 은 "meta:<mode>" 로 통일해서 1번 파일 로그와 나중에 조인하기 쉽게 함.
#
# 이 파일은 1번 predict.py 에서 만든 “하이브리드 예측 결과 목록”을
# 2차로 다시 고르는 두뇌 역할을 하는 거라서,
# - 가능한 한 입력을 버리지 말고
# - 위험/힌트/최소수익률을 여기에 모아서
# - 최종 1개 클래스를 확정하는 데 집중하도록 만들어져 있음.

from __future__ import annotations
import os, math, json
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pickle

__all__ = [
    "get_meta_prediction",
    "meta_predict",
    "maml_train_entry",
    "train_meta_learner",
    "load_meta_learner",
    "select",
]

# optional torch
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

# optional evo_meta_learner
_EVO_OK = False
try:
    from evo_meta_learner import (
        predict_evo_meta,               # (X_new, input_size, probs_stack=None) -> int or None
        aggregate_probs_for_meta        # (W,C)->(C,)
    )
    _EVO_OK = True
except Exception:
    _EVO_OK = False

# ===== 환경설정 =====
META_BASE_SUCCESS = float(os.getenv("META_BASE_SUCCESS", "0.55"))
_RET_TH = float(os.getenv("META_ER_THRESHOLD", "0.01"))
META_MIN_RETURN = float(os.getenv("META_MIN_RETURN", "0.01"))
EVO_META_AGG = os.getenv("EVO_META_AGG", "mean_var").lower()
EVO_META_VAR_GAMMA = float(os.getenv("EVO_META_VAR_GAMMA", "1.0"))
CLAMP_MAX_WIDTH = float(os.getenv("CLAMP_MAX_WIDTH", "0.10"))
META_CI_Z = float(os.getenv("META_CI_Z", "1.64"))
META_MIN_N = int(os.getenv("META_MIN_N", "30"))
CALIB_NAN_MODE = os.getenv("CALIB_NAN_MODE", "abstain").lower()  # "abstain" | "drop"

# =========================================================
# 🔁 predict.py 와 동일한 구간/단위 보정
# =========================================================
def _sanitize_range(lo: float, hi: float) -> Tuple[float, float]:
    """
    1번 predict.py 와 동일한 규칙:
    - 구간 절대값이 1을 넘으면 퍼센트로 보고 100으로 나눔
    - 실패하면 0.0으로 폴백
    """
    try:
        lo_f, hi_f = float(lo), float(hi)
        if abs(lo_f) > 1 or abs(hi_f) > 1:
            lo_f /= 100.0
            hi_f /= 100.0
        return lo_f, hi_f
    except Exception:
        return float(lo or 0.0), float(hi or 0.0)

def _sanitize_class_ranges(ranges: Optional[List[Tuple[float, float]]]) -> Optional[List[Tuple[float, float]]]:
    if not ranges:
        return None
    out = []
    for a, b in ranges:
        out.append(_sanitize_range(a, b))
    return out

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

# ========== (C) 안전 로그 유틸 ==========
def _safe_log_prediction(**kwargs):
    try:
        from logger import log_prediction  # 선택적 의존
        for k in ("rate", "return_value", "entry_price", "target_price"):
            if k in kwargs:
                kwargs[k] = float(np.nan_to_num(kwargs[k], nan=0.0, posinf=0.0, neginf=0.0))
        log_prediction(**kwargs)
    except Exception:
        info = {k: kwargs.get(k) for k in ["symbol", "strategy", "model",
                                           "predicted_class", "note", "rate", "reason", "source"]}
        print(f"[META-LOG Fallback] {info}")

# ================= (D) 안전 정규화/집계 유틸 =================
def _nan_guard(x: Any, *, fill: float = 0.0):
    try:
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=np.float64)
            arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
            return arr
        return float(np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill))
    except Exception:
        return fill

def _normalize_safe(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    v[~np.isfinite(v)] = 0.0
    v[v < 0] = 0.0
    s = float(v.sum())
    if s <= 0 or not np.isfinite(s):
        n = max(1, len(v))
        return np.ones(n, dtype=np.float64) / n
    return v / s

def _to_probs(x: Any, C_expected: Optional[int] = None) -> Optional[np.ndarray]:
    """
    predict.py 가 넘길 수 있는 여러 형태(dict, list, np.array)를 모두 수용
    """
    try:
        if x is None:
            return None
        if isinstance(x, dict):
            if not x:
                return None
            C = max(int(k) for k in x.keys()) + 1
            if C_expected is not None and C_expected != C:
                return None
            vec = np.zeros(C, dtype=np.float64)
            for k, v in x.items():
                ki = int(k)
                vec[ki] = float(v)
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

# ------------ 핵심: 그룹 출력에서 “가장 정보가 많은 확률” 고르기 ------------
def _pick_best_probs_from_group(g: Dict, C_guess: int) -> Optional[np.ndarray]:
    """
    predict.py 에서 하나의 모델에 대해 이런 필드를 줄 수 있음:
      - hybrid_probs (유사도+다양성 보정까지 된 최종형)
      - adjusted_probs (최근 클래스 빈도 보정형)
      - calib_probs / probs
      - filtered_probs
    여기서는 위 순서대로 가장 ‘진화된’ 확률을 우선 사용한다.
    """
    for key in ("hybrid_probs", "filtered_probs", "adjusted_probs", "calib_probs", "probs"):
        if key in g and g[key] is not None:
            arr = _to_probs(g[key], C_guess)
            if arr is not None:
                return arr
    # 전부 실패하면 None
    return None

# -------------- NaN/Inf 제거 + 단일 선택 집계 --------------
def _aggregate_pick_maxprob(groups_outputs: List[Dict]) -> Tuple[np.ndarray, Dict]:
    detail: Dict[str, Any] = {"picked": None, "candidates": []}
    C_guess = None
    for g in groups_outputs:
        arr = _pick_best_probs_from_group(g, None)
        if arr is not None:
            C_guess = arr.size
            break
    if C_guess is None:
        C_guess = 2

    candidates = []
    for idx, g in enumerate(groups_outputs):
        arr = _pick_best_probs_from_group(g, C_guess)
        if arr is None:
            continue
        if not np.all(np.isfinite(arr)) or np.any(arr < 0) or float(arr.sum()) <= 0:
            continue
        arr = _normalize_safe(arr)
        maxp = float(arr.max())
        val_f1 = float(g.get("val_f1")) if g.get("val_f1") is not None else float("-inf")
        val_loss = float(g.get("val_loss")) if g.get("val_loss") is not None else float("+inf")
        # predict.py 가 success_score 를 줬다면 그걸 1순위로 본다
        succ_sc = float(g.get("success_score", maxp))
        candidates.append((idx, arr, succ_sc, maxp, val_f1, val_loss, g))

    if not candidates:
        uniform = np.ones(C_guess, dtype=np.float64) / C_guess
        detail.update({"no_valid_model": True, "probs_stack_shape": [0, C_guess]})
        return uniform.astype(np.float32), detail

    # success_score → maxp → f1 → loss 순으로 정렬
    candidates.sort(key=lambda t: (-t[2], -t[3], -t[4], t[5], t[0]))
    best_idx, best_arr, best_succ, best_maxp, best_f1, best_loss, best_g = candidates[0]
    detail.update({
        "picked": int(best_idx),
        "picked_max_prob": float(best_maxp),
        "picked_success_score": float(best_succ),
        "picked_val_f1": float(best_f1) if np.isfinite(best_f1) else None,
        "picked_val_loss": float(best_loss) if np.isfinite(best_loss) else None,
        "picked_model_path": best_g.get("model_path"),
        "picked_model_type": best_g.get("model_type"),
        "valid_model_count": len(candidates),
        "no_valid_model": False
    })
    return best_arr.astype(np.float32), detail

def _aggregate_base_outputs(
    groups_outputs: List[Dict],
    class_success: Optional[Dict[int, float]] = None,
    mode: str = "avg"
) -> Tuple[np.ndarray, Dict]:
    if not groups_outputs:
        raise ValueError("groups_outputs 비어있음")
    if mode == "maxprob_pick":
        return _aggregate_pick_maxprob(groups_outputs)

    C_guess = None
    for g in groups_outputs:
        arr = _pick_best_probs_from_group(g, None)
        if arr is not None:
            C_guess = arr.size
            break
    if C_guess is None:
        C_guess = 2

    probs_mat = []
    for g in groups_outputs:
        arr = _pick_best_probs_from_group(g, C_guess)
        if arr is None:
            continue
        if not np.all(np.isfinite(arr)) or np.any(arr < 0) or float(arr.sum()) <= 0:
            continue
        probs_mat.append(_normalize_safe(arr))
    if not probs_mat:
        agg = np.ones(C_guess, dtype=np.float64) / C_guess
        detail = {"entropy": _entropy(agg), "top1": int(np.argmax(agg)),
                  "top1_prob": float(agg.max()), "margin": 0.0,
                  "probs_stack_shape": [0, C_guess], "fallback": "all_invalid_uniform",
                  "no_valid_model": True}
        return agg.astype(np.float32), detail

    probs_mat = np.stack(probs_mat, axis=0)
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
        agg = probs_mat.mean(axis=0)
        detail["fallback"] = f"unknown_agg:{mode}"

    agg = _normalize_safe(agg)
    detail["entropy"] = _entropy(agg)
    detail["top1"] = int(np.argmax(agg))
    detail["top1_prob"] = float(agg[detail["top1"]])
    top2 = float(np.partition(agg, -2)[-2]) if C >= 2 else 0.0
    detail["margin"] = float(detail["top1_prob"] - top2)
    detail["probs_stack_shape"] = list(probs_mat.shape)
    detail["no_valid_model"] = False
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
            lo_raw, hi_raw = class_ranges[c]
            lo, hi = _sanitize_range(lo_raw, hi_raw)
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
    predict.py 에서 fallback 으로 부를 수도 있는 기본 메타 선택기.
    """
    if not model_outputs_list:
        raise ValueError("❌ get_meta_prediction: 모델 출력 없음")

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
        if arr is not None and np.all(np.isfinite(arr)) and arr.sum() > 0:
            softmax_list.append(_normalize_safe(arr))
    if not softmax_list:
        avg_softmax = np.ones(C_guess, dtype=np.float64) / C_guess
    else:
        avg_softmax = _normalize_safe(np.mean(softmax_list, axis=0))

    meta_info = meta_info or {}
    # 여기서도 class_ranges 단위 보정
    cr_raw = meta_info.get("class_ranges", None)
    class_ranges = _sanitize_class_ranges(cr_raw)

    success_rate_dict = dict(meta_info.get("success_rate", {}))
    expected_return_dict = dict(meta_info.get("expected_return", {}))

    counts = _extract_counts(meta_info)
    if success_rate_dict:
        success_rate_dict = _adjust_success_rates_with_ci(success_rate_dict, counts)
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

    scores = _nan_guard(scores)
    final_pred_class = int(np.argmax(scores))

    try:
        _safe_log_prediction(
            symbol=meta_info.get("symbol","-"),
            strategy=meta_info.get("strategy","-"),
            direction="메타예측",
            entry_price=0,
            target_price=0,
            model="meta",
            model_name="meta:predicted",
            predicted_class=final_pred_class,
            label=final_pred_class,
            note=json.dumps({"meta_choice":"predicted","mode":mode}, ensure_ascii=False),
            success=True,
            reason=f"scores_entropy={_entropy(_normalize_safe(scores)):.3f}",
            rate=0.0,
            return_value=0.0,
            volatility=False,
            source="meta",
            group_id=0
        )
    except Exception:
        pass

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
        probs_list = []
        for g in groups_outputs:
            arr = _pick_best_probs_from_group(g, len(agg_probs))
            if arr is not None:
                probs_list.append(_normalize_safe(arr))
        if not probs_list:
            return None
        probs_stack = np.stack(probs_list, axis=0)
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

# ===== (G0) 심볼/전략 일치 필터 =====
def _filter_groups_by_symbol_strategy(groups_outputs: List[Dict], symbol: str, horizon: str) -> List[Dict]:
    out = []
    removed = 0
    for g in groups_outputs:
        s_ok = (str(g.get("symbol","")).upper() == str(symbol).upper()) if g.get("symbol") else True
        h_ok = (str(g.get("strategy","")) == str(horizon)) if g.get("strategy") else True
        if s_ok and h_ok:
            out.append(g)
        else:
            removed += 1
    if removed > 0:
        print(f"[META] symbol/strategy 불일치 {removed}개 제거 → {len(out)}개 유지")
    return out

# ========== (G) 단일 진입점: meta_predict(...) ==========
def meta_predict(
    symbol: str,
    horizon: str,
    groups_outputs: List[Dict],
    features: Optional["torch.Tensor"] = None,
    meta_state: Optional[Dict] = None,
    *,
    agg_mode: str = "maxprob_pick",
    use_stacking: bool = True,
    use_evo_meta: bool = True,
    log: bool = True,
    source: str = "meta",
    position_hint: Optional[Dict[str, bool]] = None,
    min_return_thr: Optional[float] = None
) -> Dict:
    meta_state = meta_state or {}

    # (G0) 심볼/전략 일치 필터
    raw_len = len(groups_outputs)
    groups_outputs = _filter_groups_by_symbol_strategy(groups_outputs, symbol, horizon)
    if not groups_outputs:
        if log:
            _safe_log_prediction(
                symbol=symbol, strategy=horizon, direction="메타예측",
                model="meta", model_name="meta:none",
                predicted_class=-1, label=-1,
                note=json.dumps({"filtered_all": True, "raw_len": raw_len}, ensure_ascii=False),
                success=False, reason="no_groups_after_filter",
                rate=0.0, return_value=0.0, source=source, group_id=0,
                entry_price=0, target_price=0, volatility=False
            )
        return {"class": -1, "probs": [], "confidence": 0.0, "margin": 0.0,
                "entropy": 0.0, "mode": "none", "detail": {"filtered_all": True, "raw_len": raw_len},
                "no_valid_model": True, "meta_choice": "none"}

    # meta_state 에 있는 class_ranges 단위 보정
    class_ranges = _sanitize_class_ranges(meta_state.get("class_ranges", None))

    class_success_raw = dict(meta_state.get("success_rate", {}))
    expected_return_raw = dict(meta_state.get("expected_return", {}))

    allow_long = bool((position_hint or {}).get("allow_long", True))
    allow_short = bool((position_hint or {}).get("allow_short", True))
    min_thr = float(min_return_thr if min_return_thr is not None else max(META_MIN_RETURN, _RET_TH))

    # 1차 집계
    agg_probs, detail = _aggregate_base_outputs(groups_outputs, class_success_raw, mode=agg_mode)
    used_mode = agg_mode
    final_class = int(np.argmax(agg_probs))
    no_valid_model = bool(detail.get("no_valid_model", False))

    counts = _extract_counts(meta_state)
    class_success_ci = _adjust_success_rates_with_ci(class_success_raw, counts) if class_success_raw else {}
    expected_return_scaled = _width_scaled_er(expected_return_raw, class_ranges, CLAMP_MAX_WIDTH) if expected_return_raw else {}

    # 힌트/최소수익률 1차 필터
    probs_masked, mask_reasons_p = _mask_by_hint_and_minret(
        agg_probs, class_ranges,
        allow_long=allow_long, allow_short=allow_short, min_return_thr=min_thr
    )
    if probs_masked.sum() > 0:
        agg_probs = _normalize_safe(probs_masked)
        detail.setdefault("filters", {})["prob_mask"] = mask_reasons_p
    else:
        detail.setdefault("filters", {})["prob_mask"] = {"_fallback": "all_zero → ignore_mask"}
        agg_probs = _normalize_safe(agg_probs)

    # evo 메타 우선 시도
    evo_choice: Optional[int] = None
    if use_evo_meta and not no_valid_model:
        evo_choice = _maybe_evo_decide(groups_outputs, agg_probs, expected_return_scaled)
        if isinstance(evo_choice, int):
            used_mode = "evo_meta"
            final_class = int(evo_choice)

    # 스태킹 시도 (evo에 안 걸렸을 때만)
    if use_stacking and used_mode != "evo_meta" and not no_valid_model:
        try:
            clf = load_meta_learner()
            if clf is not None:
                X_stack = np.concatenate(
                    [_normalize_safe(_pick_best_probs_from_group(g, len(agg_probs))) for g in groups_outputs
                     if _pick_best_probs_from_group(g, len(agg_probs)) is not None],
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

    top1 = int(np.argmax(agg_probs))
    top1p = float(agg_probs[top1])
    margin = float(top1p - float(np.partition(agg_probs, -2)[-2]) if len(agg_probs) >= 2 else top1p)
    ent = _entropy(agg_probs)

    # 성공률/ER 반영한 2차 점수
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

    scores_masked, mask_reasons_s = _mask_by_hint_and_minret(
        scores, class_ranges,
        allow_long=allow_long, allow_short=allow_short, min_return_thr=min_thr
    )
    if scores_masked.sum() > 0:
        scores = scores_masked
        detail.setdefault("filters", {})["score_mask"] = mask_reasons_s
    else:
        scores = agg_probs.astype(np.float64).copy()
        detail.setdefault("filters", {})["score_mask"] = {"_fallback": "all_zero → use_probs"}

    low_conf = (margin < 0.05) or (ent > math.log(max(2, len(agg_probs))) * 0.8)

    if all_er_below or scores.sum() == 0:
        tmp, _ = _mask_by_hint_and_minret(
            agg_probs.copy(), class_ranges,
            allow_long=allow_long, allow_short=allow_short, min_return_thr=min_thr
        )
        scores = tmp if tmp.sum() > 0 else (np.ones_like(agg_probs, dtype=np.float64) / len(agg_probs))
        low_conf = True

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

    if low_conf:
        used_mode = "rule_fallback"
        final_class = rule_choice
    else:
        if rule_choice != final_class and scores[rule_choice] >= scores[final_class] * 1.1:
            used_mode = "rule_bias"
            final_class = rule_choice

    if not (0 <= int(final_class) < len(agg_probs)):
        final_class = int(np.argmax(agg_probs))
        used_mode += "+idx_guard"

    # (G1) pick된 원본 모델 정보
    picked_model = {}
    try:
        if isinstance(detail.get("picked"), int):
            i = int(detail["picked"])
            if 0 <= i < len(groups_outputs):
                gm = groups_outputs[i]
                picked_model = {
                    "model_path": gm.get("model_path"),
                    "model_type": gm.get("model_type"),
                    "group_id": gm.get("group_id"),
                }
    except Exception:
        picked_model = {}

    result = {
        "class": int(final_class),
        "probs": np.asarray(agg_probs, dtype=np.float32).tolist(),
        "confidence": float(np.nan_to_num(max(agg_probs), nan=0.0, posinf=0.0, neginf=0.0)),
        "margin": float(np.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)),
        "entropy": float(np.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0)),
        "mode": used_mode,
        "detail": detail,
        "no_valid_model": bool(no_valid_model),
        "meta_choice": used_mode,
        "picked_model": picked_model or None,
    }

    if log:
        er_cho = 0.0
        try:
            if class_ranges and 0 <= result["class"] < len(class_ranges):
                lo, hi = class_ranges[result["class"]]
                lo, hi = _sanitize_range(lo, hi)
                er_cho = 0.5 * (float(lo) + float(hi))
        except Exception:
            pass
        sr_cho = class_success_ci.get(result["class"], None)
        note_dict = {
            "meta_choice": used_mode,
            "top1_class": result["class"],
            "confidence": result["confidence"],
            "margin": result["margin"],
            "ER_mid": float(er_cho),
            "SR_ci": (None if sr_cho is None else float(sr_cho)),
            "TH": float(_RET_TH),
            "EVO_AGG": EVO_META_AGG,
            "gamma": float(EVO_META_VAR_GAMMA),
            "CIz": float(META_CI_Z),
            "Wmax": float(CLAMP_MAX_WIDTH),
            "allow_long": bool(allow_long),
            "allow_short": bool(allow_short),
            "minER": float(min_thr),
            "no_valid": bool(no_valid_model),
            "picked_model": picked_model or None,
            "filtered_from": raw_len,
        }
        _safe_log_prediction(
            symbol=symbol,
            strategy=horizon,
            direction="메타예측",
            entry_price=0,
            target_price=0,
            model="meta",
            model_name=f"meta:{used_mode}",
            predicted_class=result["class"],
            label=result["class"],
            note=json.dumps(note_dict, ensure_ascii=False),
            success=True,
            reason=f"entropy={result['entropy']:.3f}",
            rate=0.0,
            return_value=0.0,
            volatility=False,
            source=source,
            group_id=0
        )

    print(f"[META] mode={used_mode} class={result['class']} "
          f"conf={result['confidence']:.3f} margin={result['margin']:.3f} "
          f"entropy={result['entropy']:.3f} no_valid={no_valid_model}")
    return result

# ========== (H) 후보 선택기 ==========
def select(candidates: List[Dict[str, Any]],
           profit_min: float = META_MIN_RETURN) -> Dict[str, Any]:
    """
    입력: 후보 dict 리스트.
      필수키: 'calib_prob' (float), 'expected_return_mid' (float)
    """
    if not candidates:
        return {"abstain": True, "reason": "no_candidates"}

    _clean = []
    for idx, c in enumerate(candidates):
        cp = c.get("calib_prob", None)
        er = c.get("expected_return_mid", None)
        try:
            cp = float(cp)
            er = float(er)
        except Exception:
            cp = float("nan")
            er = 0.0
        if not np.isfinite(cp):
            if CALIB_NAN_MODE == "abstain":
                return {"abstain": True, "reason": "calib_prob_nan"}
            else:
                continue
        if abs(er) < float(profit_min):
            continue
        _clean.append((idx, cp, er, c))

    if not _clean:
        return {"abstain": True, "reason": "filtered_out_by_rules"}

    _clean.sort(key=lambda t: (-t[1], -abs(t[2]), t[0]))
    best_idx, best_cp, best_er, best_obj = _clean[0]
    best = dict(best_obj)
    best["candidate_rank"] = int(1)
    best["meta_prob"] = float(best_cp)
    best["expected_return_mid"] = float(best_er)
    best["abstain"] = False
    return best
