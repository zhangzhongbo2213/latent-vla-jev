"""Tests for the terminal feature prediction contract."""
import pytest

torch = pytest.importorskip("torch")

from lwm.losses import terminal_feature_loss
from lwm.model import TerminalFeaturePredictor


def test_predictor_returns_same_feature_shape_and_finite_values():
    model = TerminalFeaturePredictor(
        vision_dim=12, text_dim=10, hidden_dim=32, heads=4,
        latent_count=4, latent_layers=1, decoder_layers=1,
    )
    current = torch.randn(2, 16, 12)
    task = torch.randn(2, 5, 10)
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    prediction = model(current, task, mask)
    assert prediction.shape == current.shape
    assert torch.isfinite(prediction).all()


def test_prediction_loss_backpropagates_to_predictor():
    model = TerminalFeaturePredictor(
        vision_dim=8, text_dim=6, hidden_dim=32, heads=4,
        latent_count=4, latent_layers=1, decoder_layers=1,
    )
    current = torch.randn(2, 4, 8)
    task = torch.randn(2, 3, 6)
    target = torch.randn_like(current)
    loss = terminal_feature_loss(model(current, task), target)["loss"]
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_model_rejects_fully_masked_instruction():
    model = TerminalFeaturePredictor(
        vision_dim=8, text_dim=6, hidden_dim=32, heads=4,
        latent_count=4, latent_layers=1, decoder_layers=1,
    )
    with pytest.raises(ValueError, match="at least one valid"):
        model(torch.randn(1, 4, 8), torch.randn(1, 3, 6), torch.zeros(1, 3, dtype=torch.bool))
