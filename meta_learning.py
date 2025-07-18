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
    maml = MAML(model, inner_lr=inner_lr, outer_lr=outer_lr, inner_steps=inner_steps)
    tasks = []

    for (X_train, y_train), (X_val, y_val) in zip(train_loader, val_loader):
        tasks.append((X_train, y_train, X_val, y_val))

    loss = maml.meta_update(tasks)
    print(f"[MAML meta-update 완료] loss={loss:.4f}")
    return loss


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
