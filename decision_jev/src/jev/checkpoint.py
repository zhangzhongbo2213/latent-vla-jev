"""Portable Decision JEV checkpoints with embedded codebook and provenance."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .model import DecisionJEV, JEVConfig
from .codebook import WholeActionCodebook


def _config_payload(config: JEVConfig) -> dict[str, Any]:
    return asdict(config)


def save_checkpoint(path: str | Path, model: DecisionJEV, codebook: WholeActionCodebook,
                    metadata: dict[str, Any], optimizer: torch.optim.Optimizer | None = None,
                    loss_config: dict[str, Any] | None = None, epoch: int = 0,
                    best: bool = False) -> None:
    if not isinstance(model, DecisionJEV) or not isinstance(codebook, WholeActionCodebook):
        raise TypeError("model and codebook must be DecisionJEV and WholeActionCodebook")
    payload = {
        "format": "decision_jev_checkpoint_v1",
        "model_config": _config_payload(model.config),
        "model_state": model.state_dict(),
        "codebook": codebook.to_payload(),
        "codebook_fingerprint": codebook.fingerprint(),
        "metadata": dict(metadata),
        "loss_config": dict(loss_config or {}),
        "epoch": int(epoch),
        "best": bool(best),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_policy(path: str | Path, device: str | torch.device = "cpu",
                expected_metadata: dict[str, Any] | None = None) -> tuple[DecisionJEV, WholeActionCodebook, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "decision_jev_checkpoint_v1":
        raise ValueError("Unsupported checkpoint format")
    config = JEVConfig(**payload["model_config"])
    model = DecisionJEV(config).to(device)
    model.load_state_dict(payload["model_state"])
    codebook = WholeActionCodebook.from_payload(payload["codebook"]).to(device)
    if payload.get("codebook_fingerprint") != codebook.fingerprint():
        raise ValueError("Checkpoint codebook fingerprint mismatch")
    metadata = dict(payload.get("metadata", {}))
    if expected_metadata:
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ValueError(f"Checkpoint metadata mismatch: {key}")
    return model, codebook, metadata


def load_training_state(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Load all fields needed to resume training, including optimizer state."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "decision_jev_checkpoint_v1":
        raise ValueError("Unsupported checkpoint format")
    return payload
