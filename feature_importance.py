import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def compute_feature_importance(model, X_val, y_val, feature_names, batch_size=32):
    model.eval()
    with torch.no_grad():
        y_pred = (model(X_val).squeeze().numpy() > 0.5).astype(int)
    base_score = accuracy_score(y_val.numpy(), y_pred)

    importances = []
    X_array = X_val.clone().numpy()
    for i, name in enumerate(feature_names):
        X_perturbed = X_array.copy()
        np.random.shuffle(X_perturbed[:, :, i])  # i번째 feature만 섞음
        X_tensor_perturbed = torch.tensor(X_perturbed, dtype=torch.float32)
        with torch.no_grad():
            y_pred_perturbed = (model(X_tensor_perturbed).squeeze().numpy() > 0.5).astype(int)
        score = accuracy_score(y_val.numpy(), y_pred_perturbed)
        importance = base_score - score
        importances.append((name, round(importance, 4)))

    importances.sort(key=lambda x: x[1], reverse=True)
    return importances

def save_feature_importance(importances, symbol, strategy, model_type):
    df = pd.DataFrame(importances, columns=["feature", "importance"])
    path = f"/persistent/logs/feature_importance_{symbol}_{strategy}_{model_type}.csv"
    df.to_csv(path, index=False)
    print(f"[LOG] Feature importance 저장 완료 → {path}")
