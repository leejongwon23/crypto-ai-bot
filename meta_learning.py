# meta_learning.py
# ------------------------------------------------------------
# 3) 메타러너 파일 (단일 진입점 + 집계 + 룰기반 폴백 + 안전 로그)
# ------------------------------------------------------------

from __future__ import annotations
import os
import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ===== (환경설정) 성공률 기본값: 성공 이력 없을 때 사용할 prior =====
META_BASE_SUCCESS = float(os.getenv("META_BASE_SUCCESS", "0.55"))

# ========== (A) MAML - 기존 유틸 유지 ==========

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
            meta_loss += self.loss_fn(logits, y_va)
        meta_loss /= max(1, len(tasks))
        self.optimizer.zero_grad()
        self.optimizer.step()
        return meta_loss.item()

def maml_train_entry(model, train_loader, val_loader, inner_lr=0.01, outer_lr=0.001, inner_steps=1):
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


# ========== (B) 스태킹형 메타러너(Scikit) ==========

import pickle

META_MODEL_PATH = "/persistent/models/meta_learner.pkl"

def train_meta_learner(model_outputs_list, true_labels):
    """
    model_outputs_list: [probs_flatten or probs_vector ...]
    true_labels: [int, ...]
    """
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

def load_meta_learner():
    if os.path.exists(META_MODEL_PATH):
        with open(META_MODEL_PATH, "rb") as f:
            clf = pickle.load(f)
        print("[✅ meta learner 로드 완료]")
        return clf
    else:
        print("[⚠️ meta learner 파일 없음 → None 반환]")
        return None


# ========== (C) 안전 로그 유틸(있으면 사용, 없으면 프린트) ==========

def _safe_log_prediction(**kwargs):
    try:
        from logger import log_prediction  # 선택적 의존
        log_prediction(**kwargs)
    except Exception:
        # 로거 없거나 실패하면 콘솔 출력
        info = {k: kwargs.get(k) for k in ["symbol", "strategy", "model",
                                           "predicted_class", "note", "rate", "reason", "source"]}
        print(f"[META-LOG Fallback] {info}")


# ========== (D) 집계·폴백 로직 ==========

def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

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
        probs = np.asarray(g["probs"], dtype=np.float32)
        probs = probs / (probs.sum() + 1e-12)  # normalize safeguard
        probs_mat.append(probs)
    probs_mat = np.stack(probs_mat, axis=0)  # (N, C)
    C = probs_mat.shape[1]

    detail = {}
    if mode == "avg":
        agg = probs_mat.mean(axis=0)

    elif mode == "weighted":
        # 클래스 성공률이 있으면 cls별 가중치로 보정 (soft weighting)
        if not class_success:
            agg = probs_mat.mean(axis=0)
        else:
            w = np.array([class_success.get(c, META_BASE_SUCCESS) for c in range(C)], dtype=np.float32)
            w = (w - w.min()) / (w.max() - w.min() + 1e-9)  # 0~1
            w = 0.5 + 0.5 * w  # 0.5~1.0로 축소
            agg = (probs_mat * w).mean(axis=0)
            detail["class_weights"] = w.tolist()

    elif mode == "maxvote":
        # 각 모델의 argmax에 대한 다수결
        votes = np.bincount(np.argmax(probs_mat, axis=1), minlength=C)
        agg = votes.astype(np.float32) / max(1, votes.sum())
        detail["votes"] = votes.tolist()

    else:
        raise ValueError(f"알 수 없는 집계 모드: {mode}")

    agg = agg / (agg.sum() + 1e-12)
    detail["entropy"] = _entropy(agg)
    detail["top1"] = int(np.argmax(agg))
    detail["top1_prob"] = float(agg[detail["top1"]])
    # margin(탑1-탑2)
    top2_prob = float(np.partition(agg, -2)[-2]) if C >= 2 else 0.0
    detail["margin"] = float(detail["top1_prob"] - top2_prob)
    return agg, detail


# ======= (NEW) 1% 미만 페널티 헬퍼 =======
_RET_TH = 0.01  # 1%

def _ret_gain(er: float) -> float:
    """
    기대수익률 절댓값 er -> 점수 가중치.
    - 1% 미만이면 0 (사실상 선택 배제)
    - 1% 이상이면 (|er|-0.01) 만큼 비례
    """
    er_abs = abs(float(er))
    return max(0.0, er_abs - _RET_TH)


def get_meta_prediction(model_outputs_list, feature_tensor=None, meta_info=None):
    """
    (유지) 단독 유틸: 성공률/수익률 고려 스코어로 최종 클래스 산출
    model_outputs_list: [{"probs": ...}, ...] 또는 [np.array, ...]
    """
    if not model_outputs_list:
        raise ValueError("❌ get_meta_prediction: 모델 출력 없음")

    # dict/array 모두 허용
    softmax_list = []
    for m in model_outputs_list:
        if isinstance(m, dict):
            if "probs" not in m:
                raise KeyError(f"'probs' 키 누락 → {m}")
            softmax_list.append(np.array(m["probs"], dtype=np.float32))
        else:
            softmax_list.append(np.array(m, dtype=np.float32))

    num_classes = len(softmax_list[0])
    avg_softmax = (np.mean(softmax_list, axis=0) + 1e-12)
    avg_softmax = avg_softmax / avg_softmax.sum()

    meta_info = meta_info or {}
    success_rate_dict = meta_info.get("success_rate", {})      # {cls: 0~1}
    expected_return_dict = meta_info.get("expected_return", {})# {cls: r}

    # -------- 점수 계산 --------
    scores = np.zeros(num_classes, dtype=np.float32)
    all_below = True  # 모든 클래스 ER<1%인지 확인

    for c in range(num_classes):
        # ✅ 성공 이력 없으면 META_BASE_SUCCESS 사용
        sr = success_rate_dict.get(c, META_BASE_SUCCESS)
        er = expected_return_dict.get(c, 0.0)       # 없으면 0 → 임계 미만으로 처리
        g  = _ret_gain(er)                          # 1% 미만이면 0
        if g > 0:
            all_below = False
        scores[c] = avg_softmax[c] * (0.5 + 0.5*sr) * g

    mode = "성공률 기반 메타" if success_rate_dict else "기본 메타 (성공률 無)"

    # 모든 클래스가 1% 미만이면 → 확률로 선택
    if all_below:
        scores = avg_softmax.copy()
        mode += " / all<1%→prob선택"

    final_pred_class = int(np.argmax(scores))
    print(f"[META] {mode} → 최종 클래스 {final_pred_class} / 점수={scores.round(4)}")
    return final_pred_class


# ========== (E) 단일 진입점: meta_predict(...) ==========

def meta_predict(
    symbol: str,
    horizon: str,
    groups_outputs: List[Dict],
    features: Optional[torch.Tensor] = None,
    meta_state: Optional[Dict] = None,
    *,
    agg_mode: str = "avg",        # "avg" | "weighted" | "maxvote"
    use_stacking: bool = True,    # 저장된 스태킹 메타러너가 있으면 사용
    log: bool = True,
    source: str = "meta"
) -> Dict:
    """
    ✅ 단일 진입점
    - 베이스 모델 출력 집계(평균/가중치/다수결) + 스태킹/룰기반 폴백
    - 1% 미만 수익률은 자동 페널티
    """
    meta_state = meta_state or {}
    class_success = meta_state.get("success_rate", {})      # {cls: rate}
    expected_return = meta_state.get("expected_return", {}) # {cls: r}

    # 1) 베이스 집계
    agg_probs, detail = _aggregate_base_outputs(groups_outputs, class_success, mode=agg_mode)

    # 2) 스태킹 시도
    used_mode = agg_mode
    final_class = int(np.argmax(agg_probs))
    if use_stacking:
        clf = None
        try:
            clf = load_meta_learner()
        except Exception as e:
            print(f"[⚠️ stacking 로드 실패] {e}")

        if clf is not None:
            try:
                X_stack = np.concatenate([np.asarray(g["probs"], dtype=np.float32).flatten()
                                          for g in groups_outputs], axis=0).reshape(1, -1)
                stacked_pred = clf.predict(X_stack)[0]
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(X_stack)[0]
                    final_class = int(stacked_pred)
                    used_mode = "stacking"
                    try:
                        if len(proba) == len(agg_probs):
                            agg_probs = proba.astype(np.float32)
                        else:
                            used_mode = "stacking(base-probs)"
                    except Exception:
                        used_mode = "stacking(base-probs)"
                else:
                    final_class = int(stacked_pred)
                    used_mode = "stacking(base-probs)"
            except Exception as e:
                print(f"[⚠️ stacking 예측 실패 → 집계 폴백] {e}")

    # 3) 룰기반 폴백(확신 낮으면 성공률/수익률 보정 + 1% 임계 적용)
    top1 = int(np.argmax(agg_probs))
    top1p = float(agg_probs[top1])
    margin = float(top1p - float(np.partition(agg_probs, -2)[-2]) if len(agg_probs) >= 2 else top1p)
    ent = _entropy(agg_probs)

    if used_mode != "stacking":
        low_conf = (margin < 0.05) or (ent > math.log(len(agg_probs)) * 0.8)

        scores = agg_probs.copy()
        for c in range(len(scores)):
            # ✅ 동일하게 META_BASE_SUCCESS 사용 + 1% 임계
            sr = class_success.get(c, META_BASE_SUCCESS)
            er = expected_return.get(c, 0.0)
            scores[c] = scores[c] * (0.5 + 0.5 * sr) * _ret_gain(er)

        if np.all([_ret_gain(expected_return.get(c, 0.0)) == 0.0 for c in range(len(scores))]):
            scores = agg_probs.copy()
            low_conf = True

        rule_choice = int(np.argmax(scores))
        if low_conf or rule_choice != final_class:
            used_mode = "rule_fallback"
            final_class = rule_choice

    # 4) 결과 구성
    result = {
        "class": int(final_class),
        "probs": agg_probs.astype(np.float32).tolist(),
        "confidence": float(max(agg_probs)),
        "margin": float(margin),
        "entropy": float(ent),
        "mode": used_mode,
        "detail": detail,
    }

    # 5) 로깅(선택)
    if log:
        er_cho = expected_return.get(result["class"], 0.0)
        sr_cho = class_success.get(result["class"], None)
        note = (f"meta:{used_mode} top1={result['class']} p={result['confidence']:.3f} "
                f"margin={result['margin']:.3f} ER={er_cho:.4f} SR={('-' if sr_cho is None else f'{sr_cho:.2f}')}")
        reason = f"entropy={result['entropy']:.3f} TH={_RET_TH:.2%}"
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
