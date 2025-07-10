import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean', class_weight=None, y_train=None, num_classes=21):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        # ✅ class_weight 동적 계산
        if y_train is not None:
            from collections import Counter
            counts = Counter(y_train)
            total = sum(counts.values())
            weights = [total / counts.get(i, 1) for i in range(num_classes)]
            self.class_weight = torch.tensor(weights, dtype=torch.float32)
        else:
            self.class_weight = class_weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weight.to(inputs.device) if self.class_weight is not None else None)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
