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
        meta_loss.backward()
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
            w = np.array([class_success.get(c, 0.5) for c in range(C)], dtype=np.float32)
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
    avg_softmax = np.mean(softmax_list, axis=0)

    success_rate_dict = meta_info.get("success_rate", {}) if meta_info else {}
    expected_return_dict = meta_info.get("expected_return", {}) if meta_info else {}
    failure_rate_dict = {c: (1.0 - success_rate_dict.get(c, 0.5)) for c in range(num_classes)}

    scores = np.zeros(num_classes, dtype=np.float32)
    if not success_rate_dict:
        stability_weight = 1.0 - np.std(softmax_list, axis=0)
        scores = avg_softmax * stability_weight
        mode = "기본 메타 (성공률 無)"
    else:
        for c in range(num_classes):
            sr = success_rate_dict.get(c, 0.5)
            fr = failure_rate_dict.get(c, 0.5)
            er = expected_return_dict.get(c, 1.0)
            scores[c] = avg_softmax[c] * (sr - fr) * abs(er)
        mode = "성공률 기반 메타"

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
    ✅ 단일 진입점(요구사항 3-1)
    - 베이스 모델 출력 집계(평균/가중치/다수결) + 스태킹/룰기반 폴백(요구사항 3-2)
    - 선택 결과/확률/신뢰도 로깅(요구사항 3-3)
    - 학습기 미존재/손상 시 폴백 경로(요구사항 3-4)

    returns:
        {
          "class": int,
          "probs": List[float],   # 최종 집계 확률 (stacking 사용 시: stacking의 확률이 없으므로 집계 확률 반환)
          "confidence": float,    # top1 prob
          "margin": float,        # top1 - top2
          "entropy": float,
          "mode": "stacking"|"weighted"|"avg"|"maxvote"|"rule_fallback",
          "detail": {...}
        }
    """
    meta_state = meta_state or {}
    class_success = meta_state.get("success_rate", {})  # {cls: rate}
    expected_return = meta_state.get("expected_return", {})

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
                # 확률이 필요한 경우: predict_proba가 있으면 margin/신뢰도 보정
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(X_stack)[0]
                    final_class = int(stacked_pred)
                    used_mode = "stacking"
                    # stacked 확률은 클래스 공간이 동일하다고 가정. 없으면 집계확률 사용.
                    try:
                        # proba 길이가 클래스 수와 다르면 집계확률 사용
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

    # 3) 룰기반 폴백(확신 낮으면 성공률/수익률로 보정)
    top1 = int(np.argmax(agg_probs))
    top1p = float(agg_probs[top1])
    margin = float(top1p - float(np.partition(agg_probs, -2)[-2]) if len(agg_probs) >= 2 else top1p)
    ent = _entropy(agg_probs)

    if used_mode != "stacking":
        # 확신이 낮다(엔트로피 높고 마진 작다)고 판단되면 성공률/수익률 점수로 재선택
        low_conf = (margin < 0.05) or (ent > math.log(len(agg_probs)) * 0.8)
        if low_conf:
            scores = agg_probs.copy()
            for c in range(len(scores)):
                sr = class_success.get(c, 0.5)
                er = abs(expected_return.get(c, 1.0))
                scores[c] = scores[c] * (0.5 + 0.5 * sr) * er
            rule_choice = int(np.argmax(scores))
            if rule_choice != final_class:
                used_mode = "rule_fallback"
                final_class = rule_choice
                # 확률 벡터는 원본 집계 확률 유지(스코어는 선택에만 사용)

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
        note = f"meta:{used_mode} top1={result['class']} p={result['confidence']:.3f} margin={result['margin']:.3f}"
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
            reason=f"entropy={result['entropy']:.3f}",
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
