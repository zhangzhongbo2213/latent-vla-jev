"""Observation-driven subtask loop; no robot I/O or hidden execution.

The controller only proposes a short action prefix. A caller acknowledges
actually executed actions before predicted progress becomes the next PT input.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from .codebook import WholeActionCodebook
from .model import DecisionJEV


@dataclass(frozen=True)
class ActionPlan:
    plan_id: int
    token_ids: torch.Tensor          # [execution_horizon], int64
    deltas: torch.Tensor             # [execution_horizon,6], metres/radians
    progress_after: torch.Tensor    # [execution_horizon]
    gripper_ids: torch.Tensor | None
    completion_probability: float
    completed: bool


class SubtaskController:
    """One controller per robot/subtask stream, default one action per observation.

    ``completion_threshold`` is a validation-set tuning parameter, not calibrated
    confidence. FT refers to the CURRENT observation. An FT candidate pauses
    action execution while consecutive fresh observations confirm completion.
    The controller cannot detect duplicate frames; the caller supplies fresh ones.
    """

    def __init__(self, model: DecisionJEV, codebook: WholeActionCodebook,
                 *, execution_horizon: int = 1, completion_threshold: float = 0.9,
                 completion_confirmations: int = 1):
        if codebook.size != model.config.codebook_size:
            raise ValueError("model vocabulary and codebook size differ")
        if not 1 <= execution_horizon <= model.config.horizon:
            raise ValueError("execution_horizon must be within prediction horizon")
        if not math.isfinite(completion_threshold) or not 0 < completion_threshold < 1:
            raise ValueError("completion_threshold must be in (0,1)")
        if completion_confirmations < 1:
            raise ValueError("completion_confirmations must be positive")
        model_device = next(model.parameters()).device
        if codebook.centers.device != model_device:
            raise ValueError("model and codebook must share a device")
        self.model = model.eval()
        self.codebook = codebook
        self.execution_horizon = execution_horizon
        self.completion_threshold = completion_threshold
        self.completion_confirmations = completion_confirmations
        self._active = False
        self._completed = False
        self._pending: ActionPlan | None = None
        self._serial = 0
        self._hits = 0
        self.progress = 0.0

    @torch.no_grad()
    def begin_subtask(self, start_features: torch.Tensor,
                      task_features: torch.Tensor, task_mask: torch.Tensor,
                      *, lwm: Callable | None = None,
                      terminal_features: torch.Tensor | None = None) -> None:
        """Cache LWM(start, text, mask) once; all tensors have batch size one.

        Exactly one of ``lwm`` or already predicted ``terminal_features`` is
        required. Foundation feature extraction remains with the caller.
        """
        if (lwm is None) == (terminal_features is None):
            raise ValueError("provide exactly one of lwm or terminal_features")
        if start_features.ndim != 3 or start_features.shape[0] != 1:
            raise ValueError("start_features must be [1,N,Dv]")
        if task_features.ndim != 3 or task_features.shape[0] != 1:
            raise ValueError("task_features must be [1,L,Dt]")
        if (task_mask.shape != task_features.shape[:2]
                or task_mask.dtype != torch.bool or not task_mask.any()):
            raise ValueError("task_mask must be bool [1,L] with a valid token")
        if lwm is not None:
            if isinstance(lwm, torch.nn.Module):
                lwm.eval()
            terminal_features = lwm(start_features, task_features, task_mask)
        if terminal_features.shape != start_features.shape:
            raise ValueError("terminal features must match start visual token layout")
        if not torch.isfinite(terminal_features).all():
            raise ValueError("terminal features must be finite")
        self._terminal = terminal_features.detach().clone()
        self._task = task_features.detach().clone()
        self._task_mask = task_mask.detach().clone()
        self._active = True
        self._completed = False
        self._pending = None
        self._hits = 0
        self.progress = 0.0
        self._serial += 1

    @torch.no_grad()
    def plan(self, current_features: torch.Tensor, proprio: torch.Tensor) -> ActionPlan:
        if not self._active:
            raise RuntimeError("begin_subtask must be called first")
        if self._completed:
            raise RuntimeError("subtask completed; request a new VLM subtask")
        if self._pending is not None:
            raise RuntimeError("acknowledge the pending plan before observing again")
        if current_features.shape != self._terminal.shape:
            raise ValueError("current features must match cached terminal shape")
        progress = current_features.new_tensor([self.progress])
        output = self.model(current_features, self._terminal, self._task,
                            proprio, progress, self._task_mask)
        completion = float(output["completion_logits"][0].sigmoid())
        if not math.isfinite(completion):
            raise FloatingPointError("nonfinite completion probability")
        self._hits = self._hits + 1 if completion >= self.completion_threshold else 0
        self._completed = self._hits >= self.completion_confirmations
        # Wait for another observation at a candidate endpoint; never run a stale tail.
        count = 0 if self._hits else self.execution_horizon
        logits = output["action_logits"][0, :count]
        if not torch.isfinite(logits).all() or not torch.isfinite(output["progress"]).all():
            raise FloatingPointError("nonfinite policy output")
        ids = logits.argmax(-1)
        gripper = output.get("gripper_logits")
        if gripper is not None and not torch.isfinite(gripper).all():
            raise FloatingPointError("nonfinite gripper output")
        self._serial += 1
        result = ActionPlan(
            plan_id=self._serial,
            token_ids=ids.clone(),
            deltas=self.codebook.decode(ids).clone(),
            progress_after=output["progress"][0, :count].clone(),
            gripper_ids=None if gripper is None else gripper[0, :count].argmax(-1),
            completion_probability=completion,
            completed=self._completed,
        )
        # Keep an internal copy so external modification cannot corrupt feedback.
        if count:
            self._pending = ActionPlan(**{
                **result.__dict__, "progress_after": result.progress_after.clone()
            })
        return result

    def acknowledge(self, plan_id: int, executed_steps: int) -> None:
        """Advance PT only for an actually executed prefix; zero means IK/rejection.

        Acknowledging any prefix discards the remainder. Reobserve/replan next.
        Predicted progress is a temporal prior; it never directly triggers FT.
        """
        if self._pending is None or self._pending.plan_id != plan_id:
            raise ValueError("unknown, stale, or already acknowledged plan")
        if not isinstance(executed_steps, int) or not 0 <= executed_steps <= len(self._pending.token_ids):
            raise ValueError("executed_steps must be a valid prefix length")
        if executed_steps:
            self.progress = float(self._pending.progress_after[executed_steps - 1])
        self._pending = None
