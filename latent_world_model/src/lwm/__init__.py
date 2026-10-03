"""Subtask-terminal latent world model."""

from .model import TerminalFeaturePredictor
from .checkpoint import load_predictor

__all__ = ["TerminalFeaturePredictor", "load_predictor"]
