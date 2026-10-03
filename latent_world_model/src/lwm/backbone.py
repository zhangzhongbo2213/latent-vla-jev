"""Adapters for frozen, externally loaded vision encoders."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class FrozenFeatureEncoder(nn.Module):
    """Wrap a loaded image encoder and expose patch tokens as ``[B, N, D]``.

    ``backbone`` must already be constructed and loaded with the desired
    checkpoint. If it returns a mapping, set ``output_key`` to the feature
    entry (for example ``"x_norm_patchtokens"``). Spatial maps ``[B,D,H,W]``
    are flattened in row-major order. CLS tokens are never added or removed
    implicitly: select the desired feature output in the upstream adapter.
    """

    def __init__(self, backbone: nn.Module, output_key: str | None = None):
        super().__init__()
        self.backbone = backbone
        self.output_key = output_key
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        # Keep a pretrained target encoder in eval mode even when its caller
        # switches the enclosing training module to train mode.
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output: Any = self.backbone(images)
        if isinstance(output, dict):
            if self.output_key is None:
                raise ValueError("output_key is required for mapping backbone outputs")
            if self.output_key not in output:
                raise KeyError(f"Backbone output has no key {self.output_key!r}")
            output = output[self.output_key]
        elif isinstance(output, (tuple, list)):
            if not output:
                raise ValueError("Backbone returned an empty tuple/list")
            output = output[-1]
        if not isinstance(output, torch.Tensor):
            raise TypeError("Backbone feature output must be a torch.Tensor")
        if output.ndim == 4:
            output = output.flatten(2).transpose(1, 2)
        if output.ndim != 3:
            raise ValueError("Expected patch tokens [B,N,D] or spatial map [B,D,H,W]")
        return output
