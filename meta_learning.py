# meta_learning.py
# ------------------------------------------------------------
# 4) 메타러너 파일 (단일 진입점 + 집계 + 스태킹 + EVO-메타 연동 + 룰기반 폴백 + 안전 로그)
# + position_hint / min_return 필터 정합 (predict.py와 일치)
# ------------------------------------------------------------

from __future__ import annotations
import os
import math
from typing import List, Dict, Optional, Tuple

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

# ===== (환경설정) 성공률 기본값: 성공 이력 없을 때 사용할 prior =====
META_BASE_SUCCESS = float(os.getenv("META_BASE_SUCCESS", "0.55"))

# 기대수익률 임계(절댓값). 미만은 패널티(=0 처리)
_RET_TH = float(os.getenv("META_ER_THRESHOLD", "0.01"))

# predict.py와 정합: 기본 최소 기대수익(절댓값) — 필요 시 인자로 덮어씀
META_MIN_RETURN = float(os.getenv("META_MIN_RETURN", "0.01"))

# EVO-메타 합성 규칙 환경변수 (1단계와 일관)
EVO_META_AGG = os.getenv("EVO_META_AGG", "mean_var").lower()   # mean | varpen | mean_var
EVO_META_VAR_GAMMA = float(os.getenv("EVO_META_VAR_GAMMA", "1.0"))

# ===== (NEW) 폭·신뢰도 보정 관련 환경변수 =====
# 클래스 범위 최대 폭(절댓값, 예: 0.10 = 10%). 폭이 이 값보다 크면 기대수익 영향도를 축소
CLAMP_MAX_WIDTH = float(os.getenv("CLAMP_MAX_WIDTH", "0.10"))
# 성공률 신뢰구간 z-score (기본 1.64 ≈ 90% CI). 하한 CI로 보정
META_CI_Z = float(os.getenv("META_CI_Z", "1.64"))
# 표본수 스무딩을 위한 베이지안 블렌딩 기준 n0
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
            """
            deepcopy 제거 / functional update
            """
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
            if len(tasks) > 0:
                meta_loss = meta_loss / float(len(tasks))
            else:
                return 0.0
            # ✅ 역전파 추가
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
    """
    model_outputs_list: [probs_flatten or probs_vector ...]
    true_labels: [int, ...]
    """
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
        log_prediction(**kwargs)
    except Exception:
        info = {k: kwargs.get(k) for k in ["symbol", "strategy", "model",
                                           "predicted_class", "note", "rate", "reason", "source"]}
        print(f"[META-LOG Fallback] {info}")


# ================= (D) 집계·폴백 기본 로직 =================
def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    s = float(v.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(v, dtype=np.float64) / max(1, len(v))
    return v / s

def _aggregate_base_outputs(
    groups_outputs: List[Dict],
    class_success: Optional[Dict[int, float]] = None,
    mode: str = "avg"
) -> Tuple[np.ndarray, Dict]:
    """
    groups_outputs: [{"probs": np.array(C,), "group_id": int, ...}, ...]
    class_success:  {cls: success_rate in [0,1]}
    mode: "avg" | "weighted" | "maxvote"
    """
    if not groups_outputs:
        raise ValueError("groups_outputs 비어있음")

    probs_mat = []
    for g in groups_outputs:
        if "probs" not in g:
            raise KeyError(f"probs 누락: {g}")
        probs = np.asarray(g["probs"], dtype=np.float64)
        probs = _normalize(probs)  # normalize safeguard
        probs_mat.append(probs)
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
        raise ValueError(f"알 수 없는 집계 모드: {mode}")

    agg = _normalize(agg)
    detail["entropy"] = _entropy(agg)
    detail["top1"] = int(np.argmax(agg))
    detail["top1_prob"] = float(agg[detail["top1"]])
    # margin(탑1-탑2)
    top2_prob = float(np.partition(agg, -2)[-2]) if C >= 2 else 0.0
    detail["margin"] = float(detail["top1_prob"] - top2_prob)
    detail["probs_stack_shape"] = list(probs_mat.shape)
    return agg.astype(np.float32), detail


# ======= (NEW) 폭·신뢰도 보정 유틸 =======

def _ret_gain(er: float) -> float:
    """
    기대수익률 절댓값 er -> 점수 가중치.
    - 임계(_RET_TH) 미만이면 0 (사실상 선택 배제)
    - 임계 이상이면 (|er|-_RET_TH) 만큼 비례
    """
    er_abs = abs(float(er))
    return max(0.0, er_abs - _RET_TH)


def _extract_counts(meta_info: Dict) -> Dict[int, int]:
    """
    성공률 표본수 추정: meta_info에 있을 법한 키들을 유연하게 수용.
    허용 키(하나라도 있으면 사용):
      - "counts"            : {cls: n}
      - "trial_counts"      : {cls: n}
      - "samples"           : {cls: n}
      - "success_total"     : {cls: (s, n)} 형태도 허용
    """
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
    """
    성공률 신뢰구간 하한으로 보정 + 표본수 부족 시 prior와 베이지안 블렌딩.
    - sr' = max(0, sr - z*sqrt(sr*(1-sr)/n))
    - n이 작으면 sr <- (n/(n+n0))*sr + (n0/(n+n0))*prior  (사전정보와 혼합)
    """
    out: Dict[int, float] = {}
    for c, sr in sr_dict.items():
        try:
            sr = float(sr)
        except Exception:
            continue
        sr = max(0.0, min(1.0, sr))
        n = int(counts.get(c, 0))
        if n >= 0:
            sr_blend = (n / (n + n0)) * sr + (n0 / (n + n0)) * prior
        else:
            sr_blend = sr
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
    """
    클래스 폭(clamp) 반영: 폭이 max_width를 초과하면 기대수익 영향도를 축소.
    - 스케일 팩터 = min(1, max_width / width)
    - class_ranges가 없으면 er 자체 사용(폭 정보가 없으니 보정하지 않음)
    """
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


# ======= (NEW) 포지션/최소 기대수익 필터 =======

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
    """
    클래스별 (lo, hi)에서 mid=(lo+hi)/2를 계산해:
      - |mid| < min_return_thr → 0
      - pos=='long' 이고 allow_long=False → 0
      - pos=='short'이고 allow_short=False → 0
    """
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
            # 문제 시 필터 미적용
            pass
    return out, reasons


# ========== (E) 단독 유틸: 성공률/수익률 고려 최종 클래스 산출 ==========
def get_meta_prediction(model_outputs_list, feature_tensor=None, meta_info=None):
    """
    (유지) 단독 유틸: 성공률/수익률 고려 스코어로 최종 클래스 산출
    + (추가) 폭·신뢰도 보정
    """
    if not model_outputs_list:
        raise ValueError("❌ get_meta_prediction: 모델 출력 없음")

    softmax_list = []
    for m in model_outputs_list:
        if isinstance(m, dict):
            if "probs" not in m:
                raise KeyError(f"'probs' 키 누락 → {m}")
            softmax_list.append(np.array(m["probs"], dtype=np.float64))
        else:
            softmax_list.append(np.array(m, dtype=np.float64))

    num_classes = len(softmax_list[0])
    avg_softmax = _normalize(np.mean(softmax_list, axis=0))

    meta_info = meta_info or {}
    success_rate_dict = dict(meta_info.get("success_rate", {}))
    expected_return_dict = dict(meta_info.get("expected_return", {}))
    class_ranges = meta_info.get("class_ranges", None)

    # (NEW) 성공률 신뢰구간 보정
    counts = _extract_counts(meta_info)
    if success_rate_dict:
        success_rate_dict = _adjust_success_rates_with_ci(success_rate_dict, counts)

    # (NEW) 폭 기반 기대수익 영향도 축소
    if expected_return_dict:
        expected_return_dict = _width_scaled_er(expected_return_dict, class_ranges, CLAMP_MAX_WIDTH)

    scores = np.zeros(num_classes, dtype=np.float64)
    all_below = True
    for c in range(num_classes):
        sr = float(success_rate_dict.get(c, META_BASE_SUCCESS))
        er = float(expected_return_dict.get(c, 0.0))
        g  = _ret_gain(er)
        if g > 0:
            all_below = False
        scores[c] = float(avg_softmax[c]) * sr * g

    mode = "성공률 기반 메타(CI/폭보정)" if success_rate_dict or expected_return_dict else "기본 메타 (성공률/ER 無)"
    if all_below:
        scores = avg_softmax.copy()
        mode += " / all<TH→prob선택"

    final_pred_class = int(np.argmax(scores))
    print(f"[META] {mode} → 최종 클래스 {final_pred_class} / 점수={np.round(scores, 4)}")
    return final_pred_class


# ================= (F) EVO-메타 연동 보조 =================
def _build_evo_meta_vector(
    agg_probs: np.ndarray,
    expected_return: Dict[int, float]
) -> np.ndarray:
    """
    1단계 학습 스키마(softmax, expected_returns, hit)를 가능한 범위에서 재현.
    - hit 정보는 실시간 예측에서 알 수 없으므로 0으로 채움.
    - 클래스 수 C에 대해 [prob_c, er_c, hit_c(=0)] 묶음 반복 → 길이 3*C
    """
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
    """
    EVO 메타 모델이 있으면 사용해 class를 제안. 실패 시 None.
    """
    if not _EVO_OK:
        return None
    try:
        probs_stack = np.stack([_normalize(np.asarray(g["probs"], dtype=np.float64)) for g in groups_outputs], axis=0)
        try:
            _ = aggregate_probs_for_meta(probs_stack, mode=EVO_META_AGG, gamma=EVO_META_VAR_GAMMA)
        except Exception:
            pass
        X_new = _build_evo_meta_vector(agg_probs, expected_return)  # shape (3*C,)
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
    # ▼▼ 추가: predict.py와 정합되는 필터 인자 ▼▼
    position_hint: Optional[Dict[str, bool]] = None,  # {"allow_long": bool, "allow_short": bool}
    min_return_thr: Optional[float] = None
) -> Dict:
    """
    ✅ 단일 진입점
    - 베이스 모델 출력 집계 + EVO-메타/스태킹/룰기반 폴백
    - 성공률/실패율/수익률을 모두 고려
    - (NEW) 클래스폭·성공률 신뢰도 보정 통합
    - (NEW) 포지션 힌트/최소 기대수익 마스킹을 스코어/확률 레벨에 적용
    """
    meta_state = meta_state or {}
    class_success_raw = dict(meta_state.get("success_rate", {}))
    expected_return_raw = dict(meta_state.get("expected_return", {}))
    class_ranges = meta_state.get("class_ranges", None)

    allow_long = bool((position_hint or {}).get("allow_long", True))
    allow_short = bool((position_hint or {}).get("allow_short", True))
    min_thr = float(min_return_thr if min_return_thr is not None else max(META_MIN_RETURN, _RET_TH))

    # (1) 집계
    agg_probs, detail = _aggregate_base_outputs(groups_outputs, class_success_raw, mode=agg_mode)
    used_mode = agg_mode
    final_class = int(np.argmax(agg_probs))

    # (NEW) 성공률 신뢰구간 보정 + 폭 보정된 기대수익
    counts = _extract_counts(meta_state)
    if class_success_raw:
        class_success_ci = _adjust_success_rates_with_ci(class_success_raw, counts)
    else:
        class_success_ci = {}

    if expected_return_raw:
        expected_return_scaled = _width_scaled_er(expected_return_raw, class_ranges, CLAMP_MAX_WIDTH)
    else:
        expected_return_scaled = {}

    # (1.5) 우선, 확률 자체에 힌트/최소 기대수익 마스크 1차 적용 (확률 기반 경로 안정화)
    probs_masked, mask_reasons_p = _mask_by_hint_and_minret(
        agg_probs, class_ranges,
        allow_long=allow_long, allow_short=allow_short, min_return_thr=min_thr
    )
    if probs_masked.sum() > 0:
        agg_probs = _normalize(probs_masked)
        detail.setdefault("filters", {})["prob_mask"] = mask_reasons_p

    # (2) EVO 메타 (가능하면 최우선 시도)
    evo_choice: Optional[int] = None
    if use_evo_meta:
        evo_choice = _maybe_evo_decide(groups_outputs, agg_probs, expected_return_scaled)
        if isinstance(evo_choice, int):
            used_mode = "evo_meta"
            final_class = int(evo_choice)

    # (3) 스태킹 (EVO 사용 안 했거나 실패 시)
    if use_stacking and used_mode != "evo_meta":
        try:
            clf = load_meta_learner()
            if clf is not None:
                X_stack = np.concatenate(
                    [np.asarray(g["probs"], dtype=np.float64).flatten() for g in groups_outputs],
                    axis=0
                ).reshape(1, -1)
                stacked_pred = clf.predict(X_stack)[0]
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(X_stack)[0]
                    final_class = int(stacked_pred)
                    used_mode = "stacking"
                    if len(proba) == len(agg_probs):
                        agg_probs = _normalize(proba.astype(np.float64)).astype(np.float32)
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

    # (4.1) 스코어에도 힌트/최소 기대수익 마스크 2차 적용 (최종 안정화)
    scores_masked, mask_reasons_s = _mask_by_hint_and_minret(
        scores, class_ranges,
        allow_long=allow_long, allow_short=allow_short, min_return_thr=min_thr
    )
    detail.setdefault("filters", {})["score_mask"] = mask_reasons_s
    scores = scores_masked

    low_conf = (margin < 0.05) or (ent > math.log(max(2, len(agg_probs))) * 0.8)

    # 기대수익률 전부 임계 미만이면 확률로 폴백 (단, 힌트 마스크는 유지)
    if all_er_below or scores.sum() == 0:
        scores = agg_probs.astype(np.float64).copy()
        scores, _ = _mask_by_hint_and_minret(
            scores, class_ranges,
            allow_long=allow_long, allow_short=allow_short, min_return_thr=min_thr
        )
        low_conf = True

    rule_choice = int(np.argmax(scores))
    detail["final_scores"] = np.round(scores, 4).tolist()
    detail["success_rate_ci"] = {int(k): float(v) for k, v in class_success_ci.items()}
    detail["expected_return_scaled"] = {int(k): float(v) for k, v in expected_return_scaled.items()}
    detail["ci_z"] = float(META_CI_Z)
    detail["clamp_max_width"] = float(CLAMP_MAX_WIDTH)
    detail["min_return_thr"] = float(min_thr)
    detail["hint_allow_long"] = bool(allow_long)
    detail["hint_allow_short"] = bool(allow_short)

    # 최종 선택 규칙:
    # - 우선순위: EVO > 스태킹 > 집계
    # - 단, 신뢰도 낮음(low_conf)일 때는 rule_choice로 보정
    if low_conf:
        used_mode = "rule_fallback"
        final_class = rule_choice
    else:
        if rule_choice != final_class:
            if scores[rule_choice] >= scores[final_class] * 1.1:
                used_mode = "rule_bias"
                final_class = rule_choice

    result = {
        "class": int(final_class),
        "probs": np.asarray(agg_probs, dtype=np.float32).tolist(),
        "confidence": float(max(agg_probs)),
        "margin": float(margin),
        "entropy": float(ent),
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
