# === focal_loss.py (FINAL, with per-class alpha & factory) ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Union

def _to_1d_tensor(x: Union[float, Sequence[float], torch.Tensor],
                  length: int,
                  device,
                  dtype) -> torch.Tensor:
    """
    x가 스칼라/시퀀스/텐서 어느 것이든 [length] 길이의 1D 텐서로 변환.
    - 길이 불일치: 잘라내거나(>length) 1.0으로 패딩(<length)
    - dtype/device 정합
    """
    if isinstance(x, torch.Tensor):
        t = x.detach().to(device=device, dtype=dtype).flatten()
    elif isinstance(x, (list, tuple)):
        t = torch.tensor(list(x), dtype=dtype, device=device).flatten()
    else:
        # 스칼라
        t = torch.full((length,), float(x), dtype=dtype, device=device)
    if t.numel() != length:
        if t.numel() > length:
            t = t[:length]
        else:
            pad = torch.ones(length - t.numel(), dtype=dtype, device=device)
            t = torch.cat([t, pad], dim=0)
    return t

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with:
      - gamma (focusing parameter)
      - alpha: scalar or per-class tensor/sequence
      - external class_weight (for CE base weight)
      - optional dynamic weight from y_train distribution
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Union[float, Sequence[float], torch.Tensor] = 0.25,
        reduction: str = "mean",
        *,
        weight: Optional[Sequence[float]] = None,        # ✅ 외부 class weight
        class_weight: Optional[Sequence[float]] = None,  # 기존 호환
        y_train: Optional[Sequence[int]] = None,         # 라벨 분포 동적 가중
        num_classes: int = 21
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha_in = alpha
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
        self.class_weight = cw              # device/dtype/길이 보정은 forward에서 수행
        self.num_classes_hint = int(num_classes)

    def _normalize_class_weight(self, n_classes: int, device, dtype):
        """클래스 수 불일치 시 자르거나 1.0 패딩. device/dtype 일치."""
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

    def _alpha_tensor(self, n_classes: int, device, dtype):
        """
        alpha가 스칼라면 [C]로 broadcast, 시퀀스/텐서면 길이 정합.
        일반적으로 sum(alpha)로 정규화하진 않음(클래스별 중요도 배분 목적).
        """
        return _to_1d_tensor(self.alpha_in, n_classes, device, dtype)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: [B, C] (logits), targets: [B] (long)
        """
        if inputs.dim() != 2:
            raise ValueError(f"FocalLoss expects inputs [B, C], got {tuple(inputs.shape)}")
        n_classes = inputs.size(1)
        device, dtype = inputs.device, inputs.dtype

        # class weight 정합
        w = self._normalize_class_weight(n_classes, device=device, dtype=dtype)

        # CE 손실(샘플별) 및 pt
        ce = F.cross_entropy(inputs, targets, reduction="none", weight=w)
        pt = torch.exp(-ce).clamp_min(1e-12)  # 안정성

        # alpha 처리: per-class → 각 샘플에 타겟 인덱스로 gather
        alpha_vec = self._alpha_tensor(n_classes, device, dtype)  # [C]
        alpha_t = alpha_vec.gather(0, targets)                    # [B]

        # Focal term
        gamma = self.gamma
        focal_term = (1.0 - pt) ** gamma

        fl = alpha_t * focal_term * ce  # [B]

        if self.reduction == "mean":
            return fl.mean()
        if self.reduction == "sum":
            return fl.sum()
        return fl

# ----------------------------
# Factory: CE or Focal 선택기
# ----------------------------
def make_criterion(
    *,
    use_focal: bool = False,
    gamma: float = 0.0,
    alpha: Union[float, Sequence[float], torch.Tensor, None] = None,
    class_weights: Optional[Sequence[float]] = None,
    y_train: Optional[Sequence[int]] = None,
    num_classes: int = 21,
    label_smoothing: float = 0.0,
    reduction: str = "mean"
) -> nn.Module:
    """
    손실 팩토리:
      - use_focal & gamma>0 → FocalLoss(α 스칼라/클래스별 모두 허용)
      - else                → CrossEntropyLoss (class_weights & label_smoothing 적용)
    """
    if use_focal and gamma and gamma > 0.0:
        alpha_eff = 0.25 if alpha is None else alpha
        return FocalLoss(
            gamma=float(gamma),
            alpha=alpha_eff,
            reduction=reduction,
            weight=class_weights,
            y_train=y_train,
            num_classes=int(num_classes),
        )
    else:
        # PyTorch CE는 label_smoothing 지원
        w = None
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(
            weight=w,
            label_smoothing=float(label_smoothing) if label_smoothing and label_smoothing > 0 else 0.0,
            reduction=reduction
        )

__all__ = ["FocalLoss", "make_criterion"]
