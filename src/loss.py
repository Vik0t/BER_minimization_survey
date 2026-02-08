import torch.nn as nn
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss для снижения BER на "проблемных" символах.
    Усиливает градиенты для редких/сложных классов.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-entropy loss
        log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        # One-hot encoding
        targets_one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)

        # Focal loss components
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting (optional)
        if self.alpha is not None:
            alpha_weight = self.alpha * targets_one_hot + (1 - self.alpha) * (1 - targets_one_hot)
            focal_weight = focal_weight * alpha_weight.sum(dim=1)

        loss = -focal_weight * torch.log(pt + 1e-8)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
