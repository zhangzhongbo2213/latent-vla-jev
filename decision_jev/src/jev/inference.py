"""End-to-end subtask inference orchestration.

This module deliberately leaves VLM implementation and robot IK outside the
package. ``VLMPlaceholder`` is the replacement point for a future VLM class;
``RobotInterface.execute_delta`` receives physical EEPose deltas and is the
replacement point for robot-specific IK, safety checks, and joint commands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import torch

from .policy import SubtaskController


@dataclass(frozen=True)
class RobotObservation:
    """One fresh robot observation supplied to the pipeline."""

    image: Any
    proprio: torch.Tensor
    raw: Any = None


@dataclass(frozen=True)
class SubtaskProposal:
    """A VLM-produced subtask. ``description`` is encoded by FeatureProvider."""

    description: str
    completion_condition: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("subtask description must be a nonempty string")


@dataclass(frozen=True)
class SubtaskResult:
    description: str
    completed: bool
    control_steps: int
    progress: float
    completion_probability: float
    reason: str


@dataclass(frozen=True)
class InferenceResult:
    completed: bool
    subtasks: tuple[SubtaskResult, ...]
    reason: str


class VLMInterface(Protocol):
    """Interface to implement when the real VLM is selected."""

    def predict_subtask(
        self, instruction: str, history: Sequence[SubtaskResult],
        observation: RobotObservation,
    ) -> SubtaskProposal | None:
        """Return the next subtask, or ``None`` when the whole task is done."""


class VLMPlaceholder:
    """Empty VLM replacement point used until a concrete VLM is integrated.

    Subclass this class or pass an object implementing ``VLMInterface`` to
    ``InferencePipeline``. It intentionally raises instead of silently making
    up a subtask, which prevents accidental robot execution with no planner.
    """

    def predict_subtask(
        self, instruction: str, history: Sequence[SubtaskResult],
        observation: RobotObservation,
    ) -> SubtaskProposal | None:
        raise NotImplementedError(
            "VLM is not integrated. Pass a concrete object implementing "
            "VLMInterface to InferencePipeline."
        )


class FeatureProvider(Protocol):
    """Frozen encoder adapter used by both LWM and JEV."""

    def encode_image(self, image: Any) -> torch.Tensor:
        """Return visual tokens as [N,Dv] or [1,N,Dv]."""

    def encode_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return text tokens and a bool mask as [L,Dt]/[L] or batched."""


class RobotInterface(Protocol):
    """Robot adapter; IK and all safety checks remain implementation-specific."""

    def observe(self) -> RobotObservation:
        """Return a fresh image and measured proprioception."""

    def execute_delta(self, delta: torch.Tensor, gripper_id: int | None = None) -> bool:
        """Run external IK and execute one delta; return False on rejection/failure."""


def _batch_visual(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3):
        raise ValueError(f"{name} must be [N,D] or [1,N,D]")
    value = value.unsqueeze(0) if value.ndim == 2 else value
    if value.shape[0] != 1 or value.shape[1] < 1 or value.shape[2] < 1:
        raise ValueError(f"{name} must have batch size one and nonempty tokens")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating point values")
    return value


def _batch_text(features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(features, torch.Tensor) or features.ndim not in (2, 3):
        raise ValueError("text features must be [L,Dt] or [1,L,Dt]")
    features = features.unsqueeze(0) if features.ndim == 2 else features
    if features.shape[0] != 1 or features.shape[1] < 1 or features.shape[2] < 1:
        raise ValueError("text features must have batch size one and nonempty tokens")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("text features must contain finite floating point values")
    if not isinstance(mask, torch.Tensor) or mask.ndim not in (1, 2):
        raise ValueError("text mask must be [L] or [1,L]")
    mask = mask.unsqueeze(0) if mask.ndim == 1 else mask
    if mask.shape != features.shape[:2] or mask.dtype != torch.bool or not mask.any():
        raise ValueError("text mask must be bool [1,L] with at least one valid token")
    return features, mask


def _batch_proprio(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim not in (1, 2):
        raise ValueError("proprio must be [Ds] or [1,Ds]")
    value = value.unsqueeze(0) if value.ndim == 1 else value
    if value.shape[0] != 1 or value.shape[1] < 1:
        raise ValueError("proprio must have batch size one and nonempty state")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("proprio must contain finite floating point values")
    return value


class InferencePipeline:
    """Compose VLM, LWM, JEV, codebook decoding, and a robot adapter.

    A subtask's terminal feature is predicted exactly once from its start
    observation and cached by ``SubtaskController``. Each control cycle reads a
    fresh observation, executes only the controller's proposed prefix, and
    acknowledges the actually executed prefix before PT is updated. The VLM is
    called again only after current-state FT confirms the subtask boundary.
    """

    def __init__(
        self, *, vlm: VLMInterface, feature_provider: FeatureProvider,
        lwm: torch.nn.Module, controller: SubtaskController,
        robot: RobotInterface, max_control_steps: int = 100,
        max_subtasks: int = 32,
    ) -> None:
        if not hasattr(vlm, "predict_subtask"):
            raise TypeError("vlm must implement predict_subtask")
        if not hasattr(feature_provider, "encode_image") or not hasattr(feature_provider, "encode_text"):
            raise TypeError("feature_provider must implement encode_image and encode_text")
        if not isinstance(lwm, torch.nn.Module) and not callable(lwm):
            raise TypeError("lwm must be a callable module")
        if not isinstance(controller, SubtaskController):
            raise TypeError("controller must be a SubtaskController")
        if not hasattr(robot, "observe") or not hasattr(robot, "execute_delta"):
            raise TypeError("robot must implement observe and execute_delta")
        if (isinstance(max_control_steps, bool)
                or not isinstance(max_control_steps, int)
                or max_control_steps < 1):
            raise ValueError("max_control_steps must be a positive integer")
        if (isinstance(max_subtasks, bool)
                or not isinstance(max_subtasks, int)
                or max_subtasks < 1):
            raise ValueError("max_subtasks must be a positive integer")
        self.vlm = vlm
        self.feature_provider = feature_provider
        self.lwm = lwm.eval() if isinstance(lwm, torch.nn.Module) else lwm
        self.controller = controller
        self.robot = robot
        self.max_control_steps = int(max_control_steps)
        self.max_subtasks = int(max_subtasks)

    @torch.no_grad()
    def _prepare_subtask(self, proposal: SubtaskProposal,
                         observation: RobotObservation) -> None:
        start = _batch_visual(self.feature_provider.encode_image(observation.image), "image features")
        task, task_mask = _batch_text(
            *self.feature_provider.encode_text(proposal.description)
        )
        with torch.no_grad():
            terminal = self.lwm(start, task, task_mask)
        if (not isinstance(terminal, torch.Tensor)
                or terminal.shape != start.shape
                or not terminal.is_floating_point()):
            raise ValueError("LWM must return terminal features with shape [1,N,Dv]")
        if not torch.isfinite(terminal).all():
            raise ValueError("LWM returned nonfinite terminal features")
        self.controller.begin_subtask(
            start, task, task_mask, terminal_features=terminal.detach()
        )

    @torch.no_grad()
    def run_subtask(self, proposal: SubtaskProposal,
                    first_observation: RobotObservation | None = None) -> SubtaskResult:
        """Run one VLM proposal until current-state FT or a safety limit."""
        observation = (
            first_observation if first_observation is not None
            else self.robot.observe()
        )
        self._prepare_subtask(proposal, observation)
        executed_cycles = 0
        last_completion = 0.0
        for cycle_index in range(self.max_control_steps):
            # ``executed_cycles`` counts plans that ran at least one action. A
            # completion candidate can intentionally produce an empty plan, so
            # use the loop index to ensure each retry gets a fresh observation.
            observation = (
                observation if cycle_index == 0 else self.robot.observe()
            )
            current = _batch_visual(self.feature_provider.encode_image(observation.image),
                                    "current image features")
            proprio = _batch_proprio(observation.proprio)
            plan = self.controller.plan(current, proprio)
            last_completion = plan.completion_probability
            if plan.completed:
                return SubtaskResult(
                    proposal.description, True, executed_cycles,
                    self.controller.progress, last_completion, "ft_confirmed",
                )
            if plan.deltas.ndim != 2 or plan.deltas.shape[-1] != 6:
                return SubtaskResult(
                    proposal.description, False, executed_cycles,
                    self.controller.progress, last_completion, "invalid_action_plan",
                )
            if not len(plan.deltas):
                # SubtaskController emits no actions while an FT candidate is
                # being confirmed across fresh observations. There is no
                # pending plan to acknowledge in this branch.
                continue
            executed = 0
            for index, delta in enumerate(plan.deltas):
                gripper = None if plan.gripper_ids is None else int(plan.gripper_ids[index].item())
                accepted = self.robot.execute_delta(delta.detach().clone(), gripper)
                if accepted is False:
                    break
                executed += 1
            self.controller.acknowledge(plan.plan_id, executed)
            if executed == 0:
                return SubtaskResult(
                    proposal.description, False, executed_cycles,
                    self.controller.progress, last_completion, "robot_rejected_action",
                )
            executed_cycles += 1
        return SubtaskResult(
            proposal.description, False, executed_cycles,
            self.controller.progress, last_completion, "control_step_limit",
        )

    @torch.no_grad()
    def run(self, instruction: str) -> InferenceResult:
        """Run the complete task; VLM remains the only unimplemented component."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a nonempty string")
        history: list[SubtaskResult] = []
        observation = self.robot.observe()
        for _ in range(self.max_subtasks):
            proposal = self.vlm.predict_subtask(instruction, tuple(history), observation)
            if proposal is None:
                return InferenceResult(True, tuple(history), "vlm_finished")
            if not isinstance(proposal, SubtaskProposal):
                raise TypeError("VLM must return SubtaskProposal or None")
            result = self.run_subtask(proposal, observation)
            history.append(result)
            if not result.completed:
                return InferenceResult(False, tuple(history), result.reason)
            observation = self.robot.observe()
        return InferenceResult(False, tuple(history), "subtask_limit")


__all__ = [
    "FeatureProvider", "InferencePipeline", "InferenceResult", "RobotInterface",
    "RobotObservation", "SubtaskProposal", "SubtaskResult", "VLMInterface",
    "VLMPlaceholder",
]
