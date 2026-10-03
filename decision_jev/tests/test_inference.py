"""Smoke tests for the VLM -> LWM -> JEV -> robot inference loop."""
import pytest

torch = pytest.importorskip("torch")

from jev.codebook import WholeActionCodebook
from jev.inference import (
    InferencePipeline,
    RobotObservation,
    SubtaskProposal,
    VLMPlaceholder,
)
from jev.model import DecisionJEV, JEVConfig
from jev.policy import SubtaskController


class FakeFeatures:
    def encode_image(self, image):
        return image

    def encode_text(self, text):
        return torch.ones(2, 3), torch.ones(2, dtype=torch.bool)


class IdentityLWM(torch.nn.Module):
    def forward(self, current, task, mask):
        return current


class FakeRobot:
    def __init__(self):
        self.observations = 0
        self.executed = []

    def observe(self):
        self.observations += 1
        return RobotObservation(torch.zeros(2, 4), torch.zeros(2))

    def execute_delta(self, delta, gripper_id=None):
        self.executed.append((delta, gripper_id))
        return True


class OneSubtaskVLM:
    def __init__(self):
        self.calls = 0

    def predict_subtask(self, instruction, history, observation):
        self.calls += 1
        return SubtaskProposal("move to the target") if self.calls == 1 else None


def make_pipeline(*, completion_bias: float, completion_confirmations: int = 1):
    model = DecisionJEV(JEVConfig(
        vision_dim=4, text_dim=3, state_dim=2, codebook_size=4,
        horizon=2, hidden_dim=8, heads=2, context_layers=1,
        decoder_layers=1,
    ))
    with torch.no_grad():
        model.completion_head.weight.zero_()
        model.completion_head.bias.fill_(completion_bias)
    centers = torch.zeros(4, 6)
    centers[1, 0] = 0.01
    centers[2, 1] = -0.01
    centers[3, 2] = 0.01
    codebook = WholeActionCodebook(centers, torch.ones(6))
    controller = SubtaskController(
        model, codebook, execution_horizon=1, completion_threshold=0.9,
        completion_confirmations=completion_confirmations,
    )
    robot = FakeRobot()
    vlm = OneSubtaskVLM()
    pipeline = InferencePipeline(
        vlm=vlm, feature_provider=FakeFeatures(), lwm=IdentityLWM(),
        controller=controller, robot=robot, max_control_steps=2,
    )
    return pipeline, vlm, robot


def test_vlm_boundary_and_ft_return_to_vlm():
    pipeline, vlm, robot = make_pipeline(completion_bias=10.0)
    result = pipeline.run("pick up the object")
    assert result.completed
    assert result.reason == "vlm_finished"
    assert len(result.subtasks) == 1
    assert result.subtasks[0].reason == "ft_confirmed"
    assert vlm.calls == 2
    assert robot.executed == []


def test_controller_executes_codebook_delta_and_stops_at_limit():
    pipeline, vlm, robot = make_pipeline(completion_bias=-10.0)
    result = pipeline.run("move the object")
    assert not result.completed
    assert result.reason == "control_step_limit"
    assert result.subtasks[0].control_steps == 2
    assert len(robot.executed) == 2
    assert robot.executed[0][0].shape == (6,)
    assert vlm.calls == 1


def test_vlm_placeholder_fails_explicitly():
    with pytest.raises(NotImplementedError, match="VLM is not integrated"):
        VLMPlaceholder().predict_subtask("task", (), RobotObservation(None, torch.zeros(2)))


def test_ft_confirmation_reobserves_without_acknowledging_empty_plan():
    pipeline, vlm, robot = make_pipeline(
        completion_bias=10.0, completion_confirmations=2,
    )
    result = pipeline.run("place the object")
    assert result.completed
    assert result.subtasks[0].reason == "ft_confirmed"
    # Initial run observation, one confirmation observation, and the
    # observation passed to the second VLM call.
    assert robot.observations >= 3
    assert vlm.calls == 2
