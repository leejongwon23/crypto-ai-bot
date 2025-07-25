# meta_learning.py

import torch
import torch.nn as nn
import torch.optim as optim

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
        ✅ [수정]
        - deepcopy 제거 → 메모리 최적화
        - functional API 기반 가중치 업데이트로 변경
        """
        adapted_params = {name: param for name, param in self.model.named_parameters()}
        for _ in range(self.inner_steps):
            logits = self.model.forward(X, params=adapted_params) if hasattr(self.model, 'forward') else self.model(X)
            loss = self.loss_fn(logits, y)
            grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)

            # ✅ inplace gradient step
            adapted_params = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(adapted_params.items(), grads)
            }

        # ✅ adapted_params 반환 → 외부에서 functional forward 호출 가능
        return adapted_params

    def meta_update(self, tasks):
        meta_loss = 0.0
        for X_train, y_train, X_val, y_val in tasks:
            adapted_params = self.adapt(X_train, y_train)
            # functional forward with adapted_params
            logits = self.model.forward(X_val, params=adapted_params) if hasattr(self.model, 'forward') else self.model(X_val)
            loss = self.loss_fn(logits, y_val)
            meta_loss += loss

        meta_loss /= len(tasks)

        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()

        return meta_loss.item()

# ✅ [추가] train.py에서 호출 가능 구조 예시
def maml_train_entry(model, train_loader, val_loader, inner_lr=0.01, outer_lr=0.001, inner_steps=1):
    from meta_learning import MAML  # 내부 정의된 클래스라면 이 임포트 필요

    try:
        maml = MAML(model, inner_lr=inner_lr, outer_lr=outer_lr, inner_steps=inner_steps)
        tasks = []

        for (X_train, y_train), (X_val, y_val) in zip(train_loader, val_loader):
            tasks.append((X_train, y_train, X_val, y_val))

        if not tasks:
            print("[MAML skip] 유효한 meta task 없음 → meta update 생략")
            return None

        loss = maml.meta_update(tasks)
        print(f"[✅ MAML meta-update 완료] task={len(tasks)}, loss={loss:.4f}")
        return loss

    except Exception as e:
        print(f"[❌ MAML 예외 발생] {e}")
        return None



import os, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

META_MODEL_PATH = "/persistent/models/meta_learner.pkl"

def train_meta_learner(model_outputs_list, true_labels):
    X = [np.array(mo).flatten() for mo in model_outputs_list]
    y = np.array(true_labels)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)

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

import numpy as np

def get_meta_prediction(models_outputs, feature_tensor=None, meta_info=None):
    """
    가장 성공 가능성이 높은 클래스를 선택

    Args:
        models_outputs (list of np.ndarray): 각 모델의 softmax 확률 벡터
        feature_tensor (np.ndarray or torch.Tensor, optional): 현재 입력 feature
        meta_info (dict, optional): 클래스별 과거 성공률 또는 도달률 정보
            - 형태 예시: {'success_rate': {0: 0.7, 1: 0.4, ..., N: 0.6}}

    Returns:
        int: 선택된 최종 클래스 인덱스
    """
    num_models = len(models_outputs)
    if num_models == 0:
        raise ValueError("모델 softmax 출력 없음")

    num_classes = len(models_outputs[0])

    # ✅ 평균 softmax 계산
    avg_softmax = np.mean(models_outputs, axis=0)

    # ✅ 메타 정보 기반 보정 (성공률, 도달률)
    scores = np.copy(avg_softmax)
    if meta_info and "success_rate" in meta_info:
        for cls in range(num_classes):
            success = meta_info["success_rate"].get(cls, 0.5)  # 기본값 0.5
            scores[cls] *= success

    # ✅ 최종 클래스 선택
    selected_class = int(np.argmax(scores))
    return selected_class

