"""Dataset and validation for cached terminal-image features."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = {
    "current_features", "terminal_features", "task_features", "task_mask",
    "success", "episode_ids",
}


class FeatureCacheDataset(Dataset):
    """Load one `.pt` feature cache and retain verified successful samples."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            self.data: dict[str, Any] = torch.load(
                self.path, map_location="cpu", weights_only=True
            )
        except TypeError:  # PyTorch versions before the weights_only argument.
            self.data = torch.load(self.path, map_location="cpu")
        if not isinstance(self.data, dict):
            raise TypeError(f"{self.path} must contain a dictionary")
        missing = REQUIRED_KEYS - self.data.keys()
        if missing:
            raise ValueError(f"{self.path} missing keys: {sorted(missing)}")
        current = self.data["current_features"]
        terminal = self.data["terminal_features"]
        task = self.data["task_features"]
        mask = self.data["task_mask"]
        success = self.data["success"]
        ids = self.data["episode_ids"]
        if not all(isinstance(x, torch.Tensor) for x in
                   (current, terminal, task, mask, success)):
            raise TypeError("All feature fields, mask, and success must be tensors")
        if current.ndim != 3 or terminal.shape != current.shape:
            raise ValueError("Current and terminal features must share [B,N,Dv]")
        if task.ndim != 3 or task.shape[0] != current.shape[0]:
            raise ValueError("Task features must have shape [B,L,Dt]")
        if mask.shape != task.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("task_mask must be bool with shape [B,L]")
        if success.shape != (current.shape[0],) or success.dtype != torch.bool:
            raise ValueError("success must be bool with shape [B]")
        if len(ids) != current.shape[0] or not all(isinstance(x, str) for x in ids):
            raise ValueError("episode_ids must contain one string per sample")
        if current.shape[0] == 0:
            raise ValueError("Feature cache is empty")
        self.indices = torch.nonzero(success, as_tuple=False).flatten().tolist()
        if not self.indices:
            raise ValueError(f"{self.path} has no verified successful terminal samples")

    @property
    def episode_ids(self) -> set[str]:
        return {self.data["episode_ids"][i] for i in self.indices}

    @property
    def vision_dim(self) -> int:
        return int(self.data["current_features"].shape[-1])

    @property
    def text_dim(self) -> int:
        return int(self.data["task_features"].shape[-1])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source = self.indices[index]
        return {
            "current_features": self.data["current_features"][source].float(),
            "terminal_features": self.data["terminal_features"][source].float(),
            "task_features": self.data["task_features"][source].float(),
            "task_mask": self.data["task_mask"][source],
        }
