# === focal_loss.py (FINAL) ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        *,
        weight: Optional[Sequence[float]] = None,   # ✅ 외부 class weight 지원
        class_weight: Optional[Sequence[float]] = None,  # 기존 호환
        y_train: Optional[Sequence[int]] = None,    # 라벨 분포 기반 동적 가중치
        num_classes: int = 21
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction

        # 우선순위: weight > class_weight > y_train 동적계산 > None
        cw = None
        if weight is not None:
            cw = torch.as_tensor(weight, dtype=torch.float32)
        elif class_weight is not None:
            cw = torch.as_tensor(class_weight, dtype=torch.float32)
        elif y_train is not None:
            from collections import Counter
            cnt = Counter(int(y) for y in y_train)
            total = sum(cnt.values())
            if total > 0:
                cw = torch.tensor(
                    [total / max(1, cnt.get(i, 0)) for i in range(int(num_classes))],
                    dtype=torch.float32
                )
        self.class_weight = cw  # device/dtype/길이 보정은 forward에서 수행

    def _normalize_class_weight(self, n_classes: int, device, dtype):
        """클래스 수 불일치 시 잘라내거나 1.0 패딩. device/dtype 일치."""
        if self.class_weight is None:
            return None
        w = self.class_weight
        if w.numel() != n_classes:
            if w.numel() > n_classes:
                w = w[:n_classes]
            else:
                pad = torch.ones(n_classes - w.numel(), dtype=w.dtype)
                w = torch.cat([w, pad], dim=0)
        return w.to(device=device, dtype=dtype)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: [B, C], targets: [B]
        """
        n_classes = inputs.size(1)
        w = self._normalize_class_weight(n_classes, device=inputs.device, dtype=inputs.dtype)

        ce = F.cross_entropy(inputs, targets, reduction="none", weight=w)
        pt = torch.exp(-ce)
        fl = self.alpha * ((1.0 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return fl.mean()
        if self.reduction == "sum":
            return fl.sum()
        return fl
