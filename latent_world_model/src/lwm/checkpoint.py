"""Checkpoint loading helpers for training and downstream JEV integration."""
from __future__ import annotations

from pathlib import Path

import torch

from .model import TerminalFeaturePredictor


def load_predictor(path: str | Path, device: str | torch.device = "cpu",
                   freeze: bool = True) -> TerminalFeaturePredictor:
    """Load a trained terminal-feature predictor for inference/JEV use."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("feature_contract") != "frozen_terminal_image_tokens_v1":
        raise ValueError("Checkpoint uses an unknown or incompatible feature contract")
    model = TerminalFeaturePredictor(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    if freeze:
        model.requires_grad_(False)
    return model
