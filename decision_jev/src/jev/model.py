"""Terminal-feature conditioned, parallel discrete robot action policy.

The policy consumes cached features from frozen vision/text encoders. It does
not generate language or call a robot controller. One action token denotes one
complete six-dimensional EEPose increment; ``horizon`` tokens denote that many
consecutive control steps.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class JEVConfig:
    vision_dim: int
    text_dim: int
    state_dim: int
    codebook_size: int = 1024
    horizon: int = 4
    hidden_dim: int = 256
    heads: int = 8
    context_layers: int = 4
    decoder_layers: int = 2
    dropout: float = 0.0
    gripper_classes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "vision_dim", "text_dim", "state_dim", "codebook_size", "horizon",
            "hidden_dim", "heads", "context_layers", "decoder_layers",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim < 4 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be >= 4 and divisible by heads")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        if (isinstance(self.gripper_classes, bool)
                or not isinstance(self.gripper_classes, int)
                or self.gripper_classes < 0 or self.gripper_classes == 1):
            raise ValueError("gripper_classes must be 0 (disabled) or >= 2")


def _positions(length: int, dim: int, reference: torch.Tensor) -> torch.Tensor:
    """Sinusoidal positions with arbitrary sequence length and hidden width."""
    indices = torch.arange(length, device=reference.device, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, dim, 2, device=reference.device, dtype=torch.float32)
        / dim
    )
    angles = indices[:, None] * frequencies[None]
    result = torch.zeros(length, dim, device=reference.device)
    result[:, 0::2] = angles.sin()
    result[:, 1::2] = angles[:, :dim // 2].cos()
    return result.to(dtype=reference.dtype)


def _spatial_positions(length: int, dim: int, reference: torch.Tensor) -> torch.Tensor:
    """Use a square patch grid when possible, otherwise a 1 x N ordered grid.

    Input features must use the same patch ordering as the LWM. Multi-camera
    or non-square grids need explicit view/grid metadata in a later extension;
    this baseline treats their flattened ordering as a single sequence.
    """
    side = math.isqrt(length)
    height, width = (side, side) if side * side == length else (1, length)
    if dim % 4:
        return _positions(length, dim, reference)
    y = _positions(height, dim // 2, reference)[:, None].expand(height, width, -1)
    x = _positions(width, dim // 2, reference)[None].expand(height, width, -1)
    return torch.cat((y, x), dim=-1).reshape(length, dim)


class DecisionJEV(nn.Module):
    """Shared multimodal context with typed parallel action/status queries.

    ``forward`` returns action logits ``[B,H,K]``, progress ``[B,H]`` for the
    states *after* executing each action, and completion logits ``[B]`` for the
    *currently observed* state. ``task_mask=True`` means a valid text token.
    The FT query cannot attend to the action queries. No action target enters
    this network. Inputs are detached; projection layers remain trainable.
    """

    def __init__(self, config: JEVConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.vision_projection = nn.Sequential(
            nn.LayerNorm(config.vision_dim), nn.Linear(config.vision_dim, d)
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(config.text_dim), nn.Linear(config.text_dim, d)
        )
        # Do not LayerNorm raw physical state: absolute magnitudes matter.
        self.state_projection = nn.Sequential(
            nn.Linear(config.state_dim, d), nn.GELU(), nn.Linear(d, d)
        )
        self.progress_projection = nn.Sequential(
            nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d)
        )
        # Current image, predicted terminal image, subtask text, state, PT.
        self.role_embeddings = nn.Parameter(torch.randn(5, d) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.heads, dim_feedforward=4 * d,
            dropout=config.dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.context_layers,
            norm=nn.LayerNorm(d), enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=config.heads, dim_feedforward=4 * d,
            dropout=config.dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.decoder_layers, norm=nn.LayerNorm(d)
        )
        self.action_queries = nn.Parameter(torch.randn(1, config.horizon, d) * 0.02)
        self.status_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.action_head = nn.Linear(d, config.codebook_size)
        self.progress_head = nn.Linear(d, 1)
        self.completion_head = nn.Linear(d, 1)
        self.gripper_head = (
            nn.Linear(d, config.gripper_classes) if config.gripper_classes else None
        )
        query_mask = torch.zeros(config.horizon + 1, config.horizon + 1, dtype=torch.bool)
        query_mask[0, 1:] = True
        self.register_buffer("query_mask", query_mask, persistent=False)
        # Transformer containers clone their layer, including initial values.
        # Initialize each matrix independently to break that symmetry.
        for stack in (self.context_encoder, self.decoder):
            for parameter in stack.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)

    def _validate_inputs(
        self, current: torch.Tensor, terminal: torch.Tensor, task: torch.Tensor,
        state: torch.Tensor, progress: torch.Tensor, mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cfg = self.config
        if current.ndim != 3 or current.shape[-1] != cfg.vision_dim:
            raise ValueError("current_features must have shape [B,N,vision_dim]")
        batch, patches, _ = current.shape
        if batch < 1 or patches < 1 or terminal.shape != current.shape:
            raise ValueError("terminal_features must match nonempty current_features [B,N,Dv]")
        if (task.ndim != 3 or task.shape[0] != batch
                or task.shape[-1] != cfg.text_dim or task.shape[1] < 1):
            raise ValueError("task_features must have nonempty shape [B,L,text_dim]")
        if state.shape != (batch, cfg.state_dim):
            raise ValueError("proprio must have shape [B,state_dim]")
        if progress.shape != (batch,):
            raise ValueError("progress must have shape [B]")
        tensors = {"current_features": current, "terminal_features": terminal,
                   "task_features": task, "proprio": state, "progress": progress}
        for name, value in tensors.items():
            if not value.is_floating_point():
                raise ValueError(f"{name} must be floating point")
            if value.device != self.role_embeddings.device:
                raise ValueError(f"{name} must be on the model device")
            if name != "task_features" and not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
        if ((progress < 0) | (progress > 1)).any():
            raise ValueError("progress must be in [0,1]")
        if mask is None:
            mask = torch.ones(task.shape[:2], dtype=torch.bool, device=task.device)
        if mask.shape != task.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("task_mask must be boolean [B,L], True for valid tokens")
        if mask.device != task.device:
            raise ValueError("task_mask must be on the feature device")
        if not mask.any(dim=1).all():
            raise ValueError("Each sample needs at least one valid task token")
        if not torch.isfinite(task[mask]).all():
            raise ValueError("Valid task_features must contain only finite values")
        return mask

    def forward(
        self, current_features: torch.Tensor, terminal_features: torch.Tensor,
        task_features: torch.Tensor, proprio: torch.Tensor, progress: torch.Tensor,
        task_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        task_mask = self._validate_inputs(
            current_features, terminal_features, task_features, proprio, progress, task_mask
        )
        dtype = self.role_embeddings.dtype
        current = self.vision_projection(current_features.detach().to(dtype=dtype))
        terminal = self.vision_projection(terminal_features.detach().to(dtype=dtype))
        # Replace padded values before projection so NaN padding cannot poison
        # attention or parameter gradients, even though keys are masked later.
        task = task_features.detach().masked_fill(~task_mask[..., None], 0).to(dtype=dtype)
        text = self.text_projection(task)
        state = self.state_projection(proprio.detach().to(dtype=dtype))[:, None]
        pt = self.progress_projection(progress.detach().to(dtype=dtype)[:, None])[:, None]
        batch, patches, _ = current.shape
        spatial = _spatial_positions(patches, self.config.hidden_dim, current)[None]
        text_position = _positions(text.shape[1], self.config.hidden_dim, text)[None]
        context = torch.cat((
            current + spatial + self.role_embeddings[0],
            terminal + spatial + self.role_embeddings[1],
            text + text_position + self.role_embeddings[2],
            state + self.role_embeddings[3], pt + self.role_embeddings[4],
        ), dim=1)
        padding_mask = torch.cat((
            torch.zeros(batch, 2 * patches, dtype=torch.bool, device=context.device),
            ~task_mask,
            torch.zeros(batch, 2, dtype=torch.bool, device=context.device),
        ), dim=1)
        memory = self.context_encoder(context, src_key_padding_mask=padding_mask)
        action_slots = self.action_queries + _positions(
            self.config.horizon, self.config.hidden_dim, self.action_queries
        )[None]
        queries = torch.cat((self.status_query, action_slots), dim=1).expand(batch, -1, -1)
        decoded = self.decoder(
            queries, memory, tgt_mask=self.query_mask,
            memory_key_padding_mask=padding_mask,
        )
        action_hidden, status_hidden = decoded[:, 1:], decoded[:, 0]
        outputs = {
            "action_logits": self.action_head(action_hidden),
            "progress": self.progress_head(action_hidden).squeeze(-1).sigmoid(),
            "completion_logits": self.completion_head(status_hidden).squeeze(-1),
        }
        if self.gripper_head is not None:
            outputs["gripper_logits"] = self.gripper_head(action_hidden)
        return outputs
