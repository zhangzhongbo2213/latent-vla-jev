"""Masked action, after-action PT and observed-state FT supervision."""
from __future__ import annotations

import math
from typing import Mapping, Protocol

import torch
from torch.nn import functional as F


class SoftActionCodebook(Protocol):
    def soft_targets(
        self, actions: torch.Tensor, *, top_k: int, temperature: float,
        hard_weight: float,
    ) -> torch.Tensor: ...


def _mask(batch: Mapping[str, torch.Tensor], name: str, shape: tuple[int, int],
          device: torch.device) -> torch.Tensor:
    value = batch[name]
    if value.shape != shape or value.dtype != torch.bool or value.device != device:
        raise ValueError(f"{name} must be boolean [B,H] on the output device")
    return value


def jev_loss(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor],
    codebook: SoftActionCodebook, *, top_k: int = 8, temperature: float = 0.1,
    hard_weight: float = 0.25, progress_weight: float = 0.2,
    completion_weight: float = 1.0, gripper_weight: float = 1.0,
    completion_pos_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute finite losses while excluding padding and terminal actions.

    Required batch keys are physical ``actions[B,H,6]``, ``action_mask[B,H]``,
    ``progress_targets[B,H]``, ``progress_mask[B,H]`` and ``completion[B]``.
    FT labels always describe the current observed state, including terminal
    samples whose action/progress masks are entirely false. Gripper targets
    ``[B,H]`` are required only when the model has a gripper head.

    Soft action targets use a neighborhood in normalized *action geometry*,
    not numeric token-ID distance. The codebook applies the hard/soft mixture
    once: ``hard_weight=0`` enables purely soft supervision. Padding values
    are never sent to the codebook or a target loss, so NaN/sentinel padding is
    safe. Targets are detached and losses average over valid entries.
    """
    for name, value in (
        ("progress_weight", progress_weight), ("completion_weight", completion_weight),
        ("gripper_weight", gripper_weight), ("completion_pos_weight", completion_pos_weight),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(hard_weight) or not 0 <= hard_weight <= 1:
        raise ValueError("hard_weight must be in [0,1]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    logits = outputs["action_logits"]
    predicted_pt = outputs["progress"]
    completion_logits = outputs["completion_logits"]
    if logits.ndim != 3 or any(size < 1 for size in logits.shape):
        raise ValueError("action_logits must have nonempty shape [B,H,K]")
    b, h, k = logits.shape
    if top_k > k:
        raise ValueError("top_k cannot exceed the codebook size")
    if predicted_pt.shape != (b, h) or completion_logits.shape != (b,):
        raise ValueError("Expected progress [B,H] and completion_logits [B]")
    for name, value in (
        ("action_logits", logits), ("progress", predicted_pt),
        ("completion_logits", completion_logits),
    ):
        if (not value.is_floating_point() or value.device != logits.device
                or not torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite floating point on the output device")
    if ((predicted_pt < 0) | (predicted_pt > 1)).any():
        raise ValueError("Predicted progress must be in [0,1]")

    action_mask = _mask(batch, "action_mask", (b, h), logits.device)
    progress_mask = _mask(batch, "progress_mask", (b, h), logits.device)
    actions = batch["actions"]
    progress_targets = batch["progress_targets"]
    completion = batch["completion"]
    for name, value, shape in (
        ("actions", actions, (b, h, 6)),
        ("progress_targets", progress_targets, (b, h)),
        ("completion", completion, (b,)),
    ):
        if value.shape != shape or value.device != logits.device:
            raise ValueError(f"{name} must have shape {shape} on the output device")
    if not actions.is_floating_point() or not progress_targets.is_floating_point():
        raise ValueError("actions and progress_targets must be floating point")
    completion_float = completion.detach().float()
    if not torch.isfinite(completion_float).all() or ((completion != 0) & (completion != 1)).any():
        raise ValueError("completion labels must be 0 or 1")

    if action_mask.any():
        valid_actions = actions.detach()[action_mask]
        if not torch.isfinite(valid_actions).all():
            raise ValueError("Valid action targets must be finite")
        with torch.no_grad():
            targets = codebook.soft_targets(
                valid_actions, top_k=top_k, temperature=temperature, hard_weight=hard_weight
            )
        if targets.shape != (valid_actions.shape[0], k):
            raise ValueError("Codebook target size must match action_logits vocabulary")
        if (not torch.isfinite(targets).all() or (targets < 0).any()
                or not torch.allclose(targets.sum(dim=-1), torch.ones_like(targets[:, 0]),
                                      atol=1e-4, rtol=1e-4)):
            raise ValueError("Codebook must return finite normalized probability targets")
        log_probabilities = F.log_softmax(logits[action_mask].float(), dim=-1)
        action_loss = -(targets.to(log_probabilities) * log_probabilities).sum(dim=-1).mean()
    else:
        action_loss = logits.float().sum() * 0.0

    if progress_mask.any():
        valid_pt = progress_targets.detach()[progress_mask]
        if (not torch.isfinite(valid_pt).all() or (valid_pt < 0).any()
                or (valid_pt > 1).any()):
            raise ValueError("Valid progress targets must be finite and in [0,1]")
        progress_loss = F.smooth_l1_loss(predicted_pt[progress_mask].float(), valid_pt.float())
    else:
        progress_loss = predicted_pt.float().sum() * 0.0

    completion_loss = F.binary_cross_entropy_with_logits(
        completion_logits.float(), completion_float,
        pos_weight=completion_logits.new_tensor(completion_pos_weight, dtype=torch.float32),
    )
    metrics = {"action": action_loss, "progress": progress_loss, "completion": completion_loss}
    total = action_loss + progress_weight * progress_loss + completion_weight * completion_loss

    if "gripper_logits" in outputs:
        gripper_logits = outputs["gripper_logits"]
        gripper_targets = batch["gripper_targets"]
        if (gripper_logits.ndim != 3 or gripper_logits.shape[:2] != (b, h)
                or gripper_logits.shape[-1] < 2 or gripper_logits.device != logits.device
                or not gripper_logits.is_floating_point()
                or not torch.isfinite(gripper_logits).all()):
            raise ValueError("gripper_logits must be finite floating point [B,H,G>=2]")
        if (gripper_targets.shape != (b, h) or gripper_targets.dtype != torch.long
                or gripper_targets.device != logits.device):
            raise ValueError("gripper_targets must be int64 [B,H] on the output device")
        if action_mask.any():
            labels = gripper_targets.detach()[action_mask]
            if (labels < 0).any() or (labels >= gripper_logits.shape[-1]).any():
                raise ValueError("Valid gripper targets must be in [0,G)")
            gripper_loss = F.cross_entropy(gripper_logits[action_mask].float(), labels)
        else:
            gripper_loss = gripper_logits.float().sum() * 0.0
        metrics["gripper"] = gripper_loss
        total = total + gripper_weight * gripper_loss

    if not torch.isfinite(total):
        raise ValueError("Nonfinite JEV loss; check predictions and loss weights")
    metrics["loss"] = total
    return metrics
