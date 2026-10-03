"""JEPA-style feature prediction losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def terminal_feature_loss(prediction: torch.Tensor, target: torch.Tensor,
                          cosine_weight: float = 1.0,
                          smooth_l1_weight: float = 0.25) -> dict[str, torch.Tensor]:
    """Dense feature loss with cosine alignment and a normalized L1 term."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Prediction and target must share [B,N,D] shape")
    pred_norm = F.normalize(prediction.float(), dim=-1, eps=1e-6)
    target_norm = F.normalize(target.detach().float(), dim=-1, eps=1e-6)
    cosine = (1.0 - (pred_norm * target_norm).sum(dim=-1)).mean()
    smooth_l1 = F.smooth_l1_loss(pred_norm, target_norm)
    total = cosine_weight * cosine + smooth_l1_weight * smooth_l1
    return {"loss": total, "cosine": cosine, "smooth_l1": smooth_l1}
