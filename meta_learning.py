# meta_learning.py
# ------------------------------------------------------------
# 4) 메타러너 파일 (단일 진입점 + 집계 + 스태킹 + EVO-메타 연동 + 룰기반 폴백 + 안전 로그)
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

# EVO-메타 합성 규칙 환경변수 (1단계와 일관)
EVO_META_AGG = os.getenv("EVO_META_AGG", "mean_var").lower()   # mean | varpen | mean_var
EVO_META_VAR_GAMMA = float(os.getenv("EVO_META_VAR_GAMMA", "1.0"))

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


# ======= (NEW) 1% 미만 페널티 헬퍼 =======
def _ret_gain(er: float) -> float:
    """
    기대수익률 절댓값 er -> 점수 가중치.
    - 임계(_RET_TH) 미만이면 0 (사실상 선택 배제)
    - 임계 이상이면 (|er|-_RET_TH) 만큼 비례
    """
    er_abs = abs(float(er))
    return max(0.0, er_abs - _RET_TH)


def get_meta_prediction(model_outputs_list, feature_tensor=None, meta_info=None):
    """
    (유지) 단독 유틸: 성공률/수익률 고려 스코어로 최종 클래스 산출
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
    success_rate_dict = meta_info.get("success_rate", {})
    expected_return_dict = meta_info.get("expected_return", {})

    scores = np.zeros(num_classes, dtype=np.float64)
    all_below = True
    for c in range(num_classes):
        sr = float(success_rate_dict.get(c, META_BASE_SUCCESS))
        er = float(expected_return_dict.get(c, 0.0))
        g  = _ret_gain(er)
        if g > 0:
            all_below = False
        scores[c] = float(avg_softmax[c]) * sr * g

    mode = "성공률 기반 메타" if success_rate_dict else "기본 메타 (성공률 無)"
    if all_below:
        scores = avg_softmax.copy()
        mode += " / all<TH→prob선택"

    final_pred_class = int(np.argmax(scores))
    print(f"[META] {mode} → 최종 클래스 {final_pred_class} / 점수={np.round(scores, 4)}")
    return final_pred_class


# ================= (E) EVO-메타 연동 보조 =================
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
        # (W,C) 확률 스택 생성
        probs_stack = np.stack([_normalize(np.asarray(g["probs"], dtype=np.float64)) for g in groups_outputs], axis=0)
        # (선택) 규칙 합성(로깅/디버그용)
        try:
            _ = aggregate_probs_for_meta(probs_stack, mode=EVO_META_AGG, gamma=EVO_META_VAR_GAMMA)
        except Exception:
            pass
        # 입력 벡터(3*C)
        X_new = _build_evo_meta_vector(agg_probs, expected_return)  # shape (3*C,)
        pred = predict_evo_meta(X_new, input_size=int(X_new.shape[0]), probs_stack=probs_stack)
        if isinstance(pred, (int, np.integer)):
            return int(pred)
    except Exception as e:
        print(f"[⚠️ EVO 메타 예측 실패] {e}")
    return None


# ========== (F) 단일 진입점: meta_predict(...) ==========
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
    source: str = "meta"
) -> Dict:
    """
    ✅ 단일 진입점
    - 베이스 모델 출력 집계 + EVO-메타/스태킹/룰기반 폴백
    - 성공률/실패율/수익률을 모두 고려
    """
    meta_state = meta_state or {}
    class_success = meta_state.get("success_rate", {})
    expected_return = meta_state.get("expected_return", {})

    # (1) 집계
    agg_probs, detail = _aggregate_base_outputs(groups_outputs, class_success, mode=agg_mode)

    used_mode = agg_mode
    final_class = int(np.argmax(agg_probs))

    # (2) EVO 메타 (가능하면 최우선 시도)
    evo_choice: Optional[int] = None
    if use_evo_meta:
        evo_choice = _maybe_evo_decide(groups_outputs, agg_probs, expected_return)
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
                        agg_probs = proba.astype(np.float32)
                else:
                    final_class = int(stacked_pred)
                    used_mode = "stacking(base-probs)"
        except Exception as e:
            print(f"[⚠️ stacking 예측 실패 → 집계 폴백] {e}")

    # (4) 신뢰도 점검 + 룰기반 보정
    top1 = int(np.argmax(agg_probs))
    top1p = float(agg_probs[top1])
    margin = float(top1p - float(np.partition(agg_probs, -2)[-2]) if len(agg_probs) >= 2 else top1p)
    ent = _entropy(agg_probs)

    # EVO/스태킹 결과와 집계 확률 간 충돌 시 판단 강화
    low_conf = (margin < 0.05) or (ent > math.log(max(2, len(agg_probs))) * 0.8)
    scores = agg_probs.astype(np.float64).copy()
    for c in range(len(scores)):
        sr = float(class_success.get(c, META_BASE_SUCCESS))
        sr = max(0.0, min(sr, 1.0))
        fr = 1.0 - sr
        er = float(expected_return.get(c, 0.0))
        g = _ret_gain(er)
        scores[c] = scores[c] * sr * g * (1.0 - 0.3*fr)

    # 기대수익률 전부 임계 미만이면 확률에 폴백
    if np.all([_ret_gain(float(expected_return.get(c, 0.0))) == 0.0 for c in range(len(scores))]):
        scores = agg_probs.astype(np.float64).copy()
        low_conf = True

    rule_choice = int(np.argmax(scores))
    detail["final_scores"] = np.round(scores, 4).tolist()

    # 최종 선택 규칙:
    # - 우선순위: EVO > 스태킹 > 집계
    # - 단, 신뢰도 낮음(low_conf)일 때는 rule_choice로 보정
    if low_conf:
        used_mode = "rule_fallback"
        final_class = rule_choice
    else:
        # 신뢰도 충분한데 스코어 보정이 다른 결과를 주면,
        # 기대수익/성공률 가중치가 월등히 높을 때만 보정
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
        er_cho = float(expected_return.get(result["class"], 0.0))
        sr_cho = class_success.get(result["class"], None)
        note = (f"meta:{used_mode} top1={result['class']} p={result['confidence']:.3f} "
                f"margin={result['margin']:.3f} ER={er_cho:.4f} "
                f"SR={('-' if sr_cho is None else f'{float(sr_cho):.2f}')} "
                f"TH={_RET_TH:.2%} EVO_AGG={EVO_META_AGG} γ={EVO_META_VAR_GAMMA}")
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
