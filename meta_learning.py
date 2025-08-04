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

def get_meta_prediction(model_outputs_list, feature_tensor=None, meta_info=None):
    """
    메타 러너: 여러 모델 예측 결과(softmax 확률 벡터)와 과거 성능·예상 수익률 정보를 이용해
    최종 예측 클래스(final_pred_class)를 결정.
    - 성공률 데이터가 없으면: softmax 확률 + 패턴 적합도 기반
    - 성공률 데이터가 있으면: 성공률 + 실패율 + softmax + 수익률 종합 평가
    """

    import numpy as np

    # ✅ 1. 유효성 검사
    if not model_outputs_list or len(model_outputs_list) == 0:
        raise ValueError("❌ get_meta_prediction: 모델 출력 없음")

    # ✅ 2. softmax 확률 벡터 추출
    softmax_list = []
    for m in model_outputs_list:
        if "probs" not in m:
            raise KeyError(f"❌ get_meta_prediction: 'probs' 키 누락 → {m}")
        softmax_list.append(np.array(m["probs"], dtype=np.float32))

    num_classes = len(softmax_list[0])
    avg_softmax = np.mean(softmax_list, axis=0)

    # ✅ 3. meta_info
    success_rate_dict = meta_info.get("success_rate", {}) if meta_info else {}
    expected_return_dict = meta_info.get("expected_return", {}) if meta_info else {}
    failure_rate_dict = {
        cls: (1.0 - success_rate_dict.get(cls, 0.5))
        for cls in range(num_classes)
    }

    # ✅ 4. 점수 계산
    scores = np.zeros(num_classes, dtype=np.float32)

    if not success_rate_dict:  
        # 📌 성공률 데이터 없을 때 → softmax + 기본 패턴 적합도 기반
        # (패턴 적합도: softmax 안정성, 표준편차 낮을수록 안정)
        stability_weight = 1.0 - np.std(softmax_list, axis=0)
        for cls in range(num_classes):
            scores[cls] = avg_softmax[cls] * stability_weight[cls]

        mode = "기본 메타 (성공률 無)"
    else:
        # 📌 성공률 데이터 있을 때 → 성공률 + 실패율 + softmax + 예상수익률 종합
        for cls in range(num_classes):
            sr = success_rate_dict.get(cls, 0.5)
            fr = failure_rate_dict.get(cls, 0.5)
            er = expected_return_dict.get(cls, 1.0)
            # 성공률 높을수록, 실패율 낮을수록, softmax 높을수록, 수익률 높을수록 가점
            scores[cls] = avg_softmax[cls] * (sr - fr) * abs(er)

        mode = "성공률 기반 메타"

    # ✅ 5. 최종 클래스 선택
    final_pred_class = int(np.argmax(scores))

    # ✅ 6. 상세 로그
    print(f"[META] {mode} → 클래스별 점수 계산:")
    for cls in range(num_classes):
        sr = success_rate_dict.get(cls, 0.0) if success_rate_dict else None
        er = expected_return_dict.get(cls, None)
        print(f"  cls {cls}: softmax={avg_softmax[cls]:.4f}, "
              f"{'성공률='+str(round(sr,2)) if sr is not None else ''} "
              f"{'예상수익률='+str(round(er,2)) if er is not None else ''} "
              f"점수={scores[cls]:.4f}")

    print(f"[META] 최종 클래스 선택 → {final_pred_class} "
          f"(점수: {scores[final_pred_class]:.4f})")

    return final_pred_class
