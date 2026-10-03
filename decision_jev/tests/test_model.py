"""Contract tests for the Decision JEV network and its masked objective."""
import pytest

torch = pytest.importorskip("torch")

from jev.codebook import WholeActionCodebook
from jev.losses import jev_loss
from jev.model import DecisionJEV, JEVConfig


def _model(**kwargs):
    config = JEVConfig(
        vision_dim=8, text_dim=6, state_dim=5, codebook_size=16,
        horizon=4, hidden_dim=32, heads=4, context_layers=1,
        decoder_layers=1, **kwargs,
    )
    return DecisionJEV(config)


def _inputs(batch=2, patches=4, text=3):
    return (
        torch.randn(batch, patches, 8), torch.randn(batch, patches, 8),
        torch.randn(batch, text, 6), torch.randn(batch, 5),
        torch.rand(batch), torch.ones(batch, text, dtype=torch.bool),
    )


def _codebook(size=16):
    centers = torch.randn(size, 6) * 0.1
    centers[0].zero_()
    return WholeActionCodebook(centers, torch.ones(6))


def test_forward_shapes_and_all_heads_backpropagate():
    model = _model(gripper_classes=3)
    inputs = _inputs()
    outputs = model(*inputs)
    assert outputs["action_logits"].shape == (2, 4, 16)
    assert outputs["progress"].shape == (2, 4)
    assert outputs["completion_logits"].shape == (2,)
    assert outputs["gripper_logits"].shape == (2, 4, 3)
    batch = {
        "actions": torch.randn(2, 4, 6) * 0.1,
        "action_mask": torch.ones(2, 4, dtype=torch.bool),
        "progress_targets": torch.rand(2, 4),
        "progress_mask": torch.ones(2, 4, dtype=torch.bool),
        "completion": torch.tensor([0., 1.]),
        "gripper_targets": torch.randint(0, 3, (2, 4)),
    }
    codebook = _codebook()
    result = jev_loss(outputs, batch, codebook, top_k=4)
    result["loss"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters()
               if parameter.requires_grad)


def test_terminal_sample_has_zero_action_loss_but_ft_is_supervised():
    model = _model()
    outputs = model(*_inputs(batch=2))
    actions = torch.randn(2, 4, 6) * 0.1
    batch = {
        "actions": actions, "action_mask": torch.tensor([[True] * 4, [False] * 4]),
        "progress_targets": torch.rand(2, 4),
        "progress_mask": torch.tensor([[True] * 4, [False] * 4]),
        "completion": torch.tensor([0., 1.]),
    }
    codebook = _codebook()
    result = jev_loss(outputs, batch, codebook, top_k=4)
    # There is one valid row in this comparison; changing masked actions must
    # not change the action objective or cause NaN propagation.
    batch["actions"][1] = float("nan")
    changed = jev_loss(outputs, batch, codebook, top_k=4)
    assert torch.isfinite(changed["loss"])
    assert torch.allclose(result["action"], changed["action"])
    assert changed["completion"].requires_grad


def test_masked_text_is_invariant_and_inputs_are_detached():
    model = _model()
    current, terminal, task, proprio, progress, mask = _inputs()
    current.requires_grad_()
    mask[:, 1] = False
    task[:, 1] = float("nan")
    out_a = model(current, terminal, task, proprio, progress, mask)
    task[:, 1] = torch.randn(2, 6) * 1000
    out_b = model(current, terminal, task, proprio, progress, mask)
    assert torch.allclose(out_a["action_logits"], out_b["action_logits"], atol=1e-6)
    assert current.grad is None


def test_invalid_progress_dimensions_and_nonfinite_values_are_rejected():
    model = _model()
    inputs = list(_inputs())
    inputs[4] = torch.tensor([1.1, 0.2])
    with pytest.raises(ValueError, match="progress must be in"):
        model(*inputs)
    inputs = list(_inputs())
    inputs[2] = torch.randn(2, 3, 7)
    with pytest.raises(ValueError, match="task_features"):
        model(*inputs)


def test_target_features_do_not_receive_hidden_gradient():
    model = _model()
    inputs = list(_inputs())
    inputs[1].requires_grad_()
    out = model(*inputs)
    out["completion_logits"].sum().backward()
    assert inputs[1].grad is None
