"""Perceiver-style predictor for terminal-image visual features."""
from __future__ import annotations

import math

import torch
from torch import nn


def _sinusoidal_positions(length: int, dim: int, device: torch.device,
                          dtype: torch.dtype) -> torch.Tensor:
    """Return deterministic 1-D positions, with no fixed token-count limit."""
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    even_dims = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    scale = torch.exp(-math.log(10000.0) * even_dims / max(dim, 1))
    angles = positions * scale.unsqueeze(0)
    result = torch.zeros(length, dim, device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(angles)
    if dim > 1:
        result[:, 1::2] = torch.cos(angles[:, :result[:, 1::2].shape[1]])
    return result.to(dtype=dtype)


def _spatial_positions(length: int, dim: int, device: torch.device,
                       dtype: torch.dtype,
                       grid_shape: tuple[int, int] | None = None) -> torch.Tensor:
    if grid_shape is None:
        side = math.isqrt(length)
        grid_shape = (side, side) if side * side == length else (1, length)
    height, width = grid_shape
    if height < 1 or width < 1 or height * width != length:
        raise ValueError("grid_shape must contain positive dimensions with H*W=N")
    if dim % 4:
        return _sinusoidal_positions(length, dim, device, dtype)
    y = _sinusoidal_positions(height, dim // 2, device, torch.float32)
    x = _sinusoidal_positions(width, dim // 2, device, torch.float32)
    y = y[:, None, :].expand(height, width, dim // 2)
    x = x[None, :, :].expand(height, width, dim // 2)
    return torch.cat([y, x], dim=-1).reshape(length, dim).to(dtype=dtype)


class PerceiverBlock(nn.Module):
    """Latent self-attention block with a feed-forward residual."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        x = x + self.self_attention(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TerminalDecoderBlock(nn.Module):
    """Patch-aligned queries attend to the compressed task-conditioned memory."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.cross_query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(queries)
        queries = queries + self.self_attention(q, q, q, need_weights=False)[0]
        queries = queries + self.cross_attention(
            self.cross_query_norm(queries), self.memory_norm(memory),
            self.memory_norm(memory), need_weights=False
        )[0]
        return queries + self.mlp(self.mlp_norm(queries))


class TerminalFeaturePredictor(nn.Module):
    """Predict terminal-image tokens in the frozen encoder's feature space.

    Args:
        vision_dim: Feature dimension ``Dv`` of the frozen vision encoder.
        text_dim: Feature dimension ``Dt`` of the frozen text encoder.
        hidden_dim: Shared Transformer width; must be divisible by ``heads``.
        heads: Attention heads.
        latent_count: Number of learned Perceiver memory tokens.
        latent_layers: Number of latent self-attention blocks.
        decoder_layers: Number of terminal patch decoder blocks.

    Inputs are current image features ``[B,N,Dv]``, text tokens ``[B,L,Dt]``,
    and an optional boolean text mask ``[B,L]`` where ``True`` means valid.
    Output shape is exactly ``[B,N,Dv]``. Each output index predicts the
    corresponding spatial token of the terminal image representation.
    """

    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int = 256,
                 heads: int = 8, latent_count: int = 16, latent_layers: int = 4,
                 decoder_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if min(vision_dim, text_dim, hidden_dim, heads, latent_count,
               latent_layers, decoder_layers) < 1:
            raise ValueError("Model dimensions and layer counts must be positive")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4 for positional encoding")

        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.vision_projection = nn.Sequential(
            nn.LayerNorm(vision_dim), nn.Linear(vision_dim, hidden_dim)
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim)
        )
        self.latents = nn.Parameter(torch.randn(1, latent_count, hidden_dim) * 0.02)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.context_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.latent_blocks = nn.ModuleList([
            PerceiverBlock(hidden_dim, heads, dropout=dropout)
            for _ in range(latent_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            TerminalDecoderBlock(hidden_dim, heads, dropout=dropout)
            for _ in range(decoder_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vision_dim)
        # Residual prediction starts as an identity map in feature space.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, current_features: torch.Tensor,
                task_features: torch.Tensor,
                task_mask: torch.Tensor | None = None,
                grid_shape: tuple[int, int] | None = None) -> torch.Tensor:
        if current_features.ndim != 3 or task_features.ndim != 3:
            raise ValueError("Expected current [B,N,Dv] and task [B,L,Dt] tokens")
        batch, num_visual, vision_dim = current_features.shape
        if vision_dim != self.vision_dim:
            raise ValueError(f"Expected vision_dim={self.vision_dim}, got {vision_dim}")
        if task_features.shape[0] != batch or task_features.shape[-1] != self.text_dim:
            raise ValueError("Task token batch or feature dimension does not match")
        if task_features.shape[1] < 1 or num_visual < 1:
            raise ValueError("Visual and task token sequences must be nonempty")
        if task_mask is not None:
            if task_mask.shape != task_features.shape[:2]:
                raise ValueError("task_mask must have shape [B,L]")
            task_mask = task_mask.to(device=task_features.device, dtype=torch.bool)
            if (~task_mask).all(dim=1).any():
                raise ValueError("Each sample must have at least one valid task token")

        vision = self.vision_projection(current_features)
        text = self.text_projection(task_features)
        position = _spatial_positions(
            num_visual, self.hidden_dim, vision.device, vision.dtype, grid_shape
        ).unsqueeze(0)
        vision = vision + position
        context = torch.cat([vision, text], dim=1)
        padding_mask = None
        if task_mask is not None:
            visual_mask = torch.zeros(
                batch, num_visual, dtype=torch.bool, device=task_mask.device
            )
            padding_mask = torch.cat([visual_mask, ~task_mask], dim=1)

        memory = self.latents.expand(batch, -1, -1)
        memory = memory + self.context_attention(
            self.context_norm(memory), context, context,
            key_padding_mask=padding_mask, need_weights=False
        )[0]
        for block in self.latent_blocks:
            memory = block(memory)

        # Patch-aligned queries retain the current image's spatial ordering.
        queries = self.vision_projection(current_features) + position
        for block in self.decoder_blocks:
            queries = block(queries, memory)
        delta = self.output_projection(self.output_norm(queries))
        return current_features + delta
