"""One token per complete 6D end-effector increment, fitted on training data.

Physical actions are ``[dx, dy, dz, rx, ry, rz]`` in metres and radians.
The last three coordinates are a rotation vector, NOT Euler-angle differences.
For ``frame='body'`` use p' = p + R @ dp, R' = R @ Exp(dr); for
``frame='world'`` use p' = p + dp, R' = Exp(dr) @ R. Each token covers exactly
``control_dt`` seconds. Datasets must agree on these conventions before fitting.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


_DIM = 6
_FORMAT_VERSION = 1
_DISTANCE_BATCH_SIZE = 4096


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _check_actions(actions: Tensor, *, matrix: bool = False) -> None:
    if not isinstance(actions, Tensor):
        raise TypeError("actions must be a torch.Tensor")
    if actions.ndim < 1 or actions.shape[-1] != _DIM:
        raise ValueError("actions must have shape [..., 6]")
    if matrix and actions.ndim != 2:
        raise ValueError("fit requires actions with shape [M, 6]")
    if actions.numel() == 0:
        raise ValueError("actions must not be empty")
    if not torch.is_floating_point(actions) or not torch.isfinite(actions).all():
        raise ValueError("actions must contain finite floating-point values")


def _squared_distances(points: Tensor, centers: Tensor) -> Tensor:
    # Accumulating six 2D differences avoids an [M, K, 6] allocation and the
    # cancellation of ||x||² + ||c||² - 2*x*c near a no-op or a close neighbour.
    distances = (points[:, :1] - centers[:, 0].unsqueeze(0)).square()
    for dim in range(1, _DIM):
        distances.add_((points[:, dim : dim + 1] - centers[:, dim].unsqueeze(0)).square())
    if not torch.isfinite(distances).all():
        raise ValueError("action distances overflowed; inspect units and scale")
    return distances


def _nearest(points: Tensor, centers: Tensor, batch_size: int) -> tuple[Tensor, Tensor]:
    indices, errors = [], []
    for chunk in points.split(batch_size):
        error, index = _squared_distances(chunk, centers).min(dim=-1)
        indices.append(index)
        errors.append(error)
    return torch.cat(indices), torch.cat(errors)


def _rotation_angle_error(actual: Tensor, reconstructed: Tensor) -> Tensor:
    """SO(3) geodesic error via unit quaternions, including the zero-angle case."""
    def quaternion(rotvec: Tensor) -> Tensor:
        theta = rotvec.norm(dim=-1, keepdim=True)
        vector = 0.5 * torch.sinc(theta / (2.0 * math.pi)) * rotvec
        return torch.cat((torch.cos(theta / 2.0), vector), dim=-1)

    left, right = quaternion(actual), quaternion(reconstructed)
    # q and -q represent the same rotation. atan2 is more accurate than acos
    # close to zero angular error.
    sign = torch.where((left * right).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    right = right * sign
    return 4.0 * torch.atan2((left - right).norm(dim=-1), (left + right).norm(dim=-1))


class WholeActionCodebook(nn.Module):
    """Fixed codebook; every ID decodes to one complete physical 6D delta.

    ``centers`` are normalized vectors, ``scale`` maps them back to physical
    units, and center 0 must be exactly zero. No mean subtraction or clipping
    occurs. Buffers move with ``.to(device)`` and serialize with state_dict.
    Learn this object once on the training split, then freeze and reuse it for
    validation, JEV training, and deployment.
    """

    def __init__(self, centers: Tensor, scale: Tensor, control_dt: float = 0.1,
                 frame: str = "body") -> None:
        super().__init__()
        if not isinstance(centers, Tensor) or centers.ndim != 2 or centers.shape[1] != _DIM:
            raise ValueError("centers must have shape [K, 6]")
        if centers.shape[0] < 2 or not torch.is_floating_point(centers):
            raise ValueError("at least two floating-point centers are required")
        if not torch.isfinite(centers).all() or torch.count_nonzero(centers[0]).item() != 0:
            raise ValueError("centers must be finite and center 0 must be exactly zero")
        if not isinstance(scale, Tensor) or scale.shape != (_DIM,):
            raise ValueError("scale must have shape [6]")
        if not torch.is_floating_point(scale) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("scale must contain finite positive floating-point values")
        if isinstance(control_dt, bool) or not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        if frame not in {"body", "world"}:
            raise ValueError("frame must be 'body' or 'world'")
        centers = centers.detach().to(dtype=torch.float32).clone()
        scale = scale.detach().to(device=centers.device, dtype=torch.float32).clone()
        if not torch.isfinite(centers).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("centers and scale must be representable in float32")
        if not torch.isfinite(centers * scale).all():
            raise ValueError("decoded centers must be finite in physical units")
        self.register_buffer("centers", centers)
        self.register_buffer("scale", scale)
        self.control_dt = float(control_dt)
        self.frame = frame

    @property
    def size(self) -> int:
        return self.centers.shape[0]

    def _normalize(self, actions: Tensor) -> Tensor:
        _check_actions(actions)
        normalized = actions.to(device=self.centers.device, dtype=torch.float32) / self.scale.float()
        if not torch.isfinite(normalized).all():
            raise ValueError("normalized actions overflowed; inspect units and scale")
        return normalized.reshape(-1, _DIM)

    @torch.no_grad()
    def encode(self, actions: Tensor) -> Tensor:
        """Nearest center IDs, with shape ``actions.shape[:-1]`` on this device."""
        ids, _ = _nearest(self._normalize(actions), self.centers.float(), _DISTANCE_BATCH_SIZE)
        return ids.reshape(actions.shape[:-1])

    @torch.no_grad()
    def decode(self, ids: Tensor) -> Tensor:
        """Look up complete physical increments, shape ``[*ids.shape, 6]``."""
        if not isinstance(ids, Tensor) or ids.dtype not in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        }:
            raise ValueError("token IDs must be an integer tensor")
        if ids.numel() and ((ids < 0).any() or (ids >= self.size).any()):
            raise ValueError(f"token IDs must lie in [0, {self.size - 1}]")
        return self.centers[ids.to(device=self.centers.device, dtype=torch.long)] * self.scale

    @torch.no_grad()
    def soft_targets(self, actions: Tensor, top_k: int = 8,
                     temperature: float = 0.1, hard_weight: float = 0.25) -> Tensor:
        """Local distance-aware targets, with optional nearest-ID supervision.

        On the closest ``top_k`` codes, q ∝ exp(-d² / (2*temperature²)).
        Return ``(1-hard_weight)*q + hard_weight*one_hot(nearest)``.
        Temperature is in normalized-action distance units, not token IDs.
        ``hard_weight=0`` supports fully soft supervision. Output is dense
        ``[..., K]``; distance workspaces are bounded by 4096*K elements.
        """
        _positive_integer(top_k, "top_k")
        if top_k > self.size:
            raise ValueError("top_k must not exceed codebook size")
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(hard_weight) or not 0 <= hard_weight <= 1:
            raise ValueError("hard_weight must be between 0 and 1")
        points = self._normalize(actions)
        results = []
        for chunk in points.split(_DISTANCE_BATCH_SIZE):
            d2, ids = _squared_distances(chunk, self.centers.float()).topk(top_k, largest=False)
            # Subtract the minimum before temperature scaling, avoiding all
            # logits becoming -inf when an action is far outside the codebook.
            relative = d2 - d2[:, :1]
            logits = -relative / (2.0 * temperature * temperature)
            if torch.isnan(logits).any():
                raise ValueError("temperature is too small for finite soft targets")
            weights = torch.softmax(logits, dim=-1) * (1.0 - hard_weight)
            weights[:, 0] += hard_weight
            target = torch.zeros((len(chunk), self.size), device=chunk.device, dtype=torch.float32)
            target.scatter_(1, ids, weights)
            results.append(target)
        return torch.cat(results).reshape(*actions.shape[:-1], self.size)

    @classmethod
    @torch.no_grad()
    def fit(cls, actions: Tensor, size: int = 1024, iterations: int = 30,
            seed: int = 0, scale: Tensor | None = None, control_dt: float = 0.1,
            frame: str = "body", batch_size: int = 4096) -> WholeActionCodebook:
        """CPU K-means++ and Lloyd updates with an immutable zero center.

        Fit ONLY on training-split actions. Initialization uses the supplied
        seed and the complete data distribution; memory for pairwise distances
        is bounded by ``batch_size * size``. Empty clusters are reseeded using
        farthest observations. Default scale is the absolute 99th percentile
        pooled across xyz and separately across rotation-vector coordinates,
        with a 1e-6 numerical floor. This scales data without clipping it.
        Sampling/reweighting rare actions, if desired, is the caller's choice.
        """
        _check_actions(actions, matrix=True)
        for name, value in (("size", size), ("iterations", iterations), ("batch_size", batch_size)):
            _positive_integer(value, name)
        if size < 2:
            raise ValueError("size must be at least 2, including the zero token")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        data = actions.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(data).all():
            raise ValueError("training actions must be representable in float32")
        if scale is None:
            translation_scale = torch.quantile(data[:, :3].abs().reshape(-1), 0.99).clamp_min(1e-6)
            rotation_scale = torch.quantile(data[:, 3:].abs().reshape(-1), 0.99).clamp_min(1e-6)
            scale = torch.cat((translation_scale.repeat(3), rotation_scale.repeat(3)))
        else:
            scale = scale.detach().to(device="cpu", dtype=torch.float32)
        # Centralize scale/convention validation before any expensive fitting.
        instance = cls(torch.zeros(size, _DIM), scale, control_dt, frame)
        points = instance._normalize(data)
        nonzero = points[torch.count_nonzero(points, dim=-1) > 0]
        if len(nonzero) == 0:
            raise ValueError("cannot fit a movement codebook to all-zero actions")
        if len(nonzero) < size - 1 or len(torch.unique(nonzero, dim=0)) < size - 1:
            raise ValueError(f"need at least {size - 1} distinct nonzero training actions for size={size}")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        centers = instance.centers
        nearest_error = points.square().sum(dim=-1)
        if not torch.isfinite(nearest_error).all():
            raise ValueError("normalized actions overflowed during initialization")
        for index in range(1, size):
            maximum = nearest_error.max()
            if maximum <= 0:
                raise ValueError("not enough numerically distinct actions for the requested size")
            chosen = torch.multinomial(nearest_error / maximum, 1, generator=generator).item()
            centers[index].copy_(points[chosen])
            # One point per center; no M*K allocation during initialization.
            for start in range(0, len(points), batch_size):
                error = (points[start : start + batch_size] - centers[index]).square().sum(-1)
                view = nearest_error[start : start + batch_size]
                view.copy_(torch.minimum(view, error))

        for _ in range(iterations):
            labels, errors = _nearest(points, centers, batch_size)
            counts = torch.bincount(labels, minlength=size)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, labels, points)
            updated = sums / counts.clamp_min(1).unsqueeze(-1)
            updated[0].zero_()
            empty = torch.nonzero(counts[1:] == 0).flatten() + 1
            if len(empty):
                # Distances to the newly updated centers keep multiple repairs
                # from choosing the same observation or duplicate rows.
                _, repair_error = _nearest(points, updated[counts > 0], batch_size)
                repair_error = torch.minimum(repair_error, points.square().sum(-1))
                for index in empty.tolist():
                    chosen = repair_error.argmax().item()
                    if repair_error[chosen] <= 0:
                        raise ValueError("empty clusters could not be reseeded; reduce codebook size")
                    updated[index].copy_(points[chosen])
                    for start in range(0, len(points), batch_size):
                        distance = (points[start : start + batch_size] - updated[index]).square().sum(-1)
                        view = repair_error[start : start + batch_size]
                        view.copy_(torch.minimum(view, distance))
            shift = (centers - updated).abs().max().item()
            centers.copy_(updated)
            if shift <= 1e-6:
                break
        centers[0].zero_()
        return cls(centers, scale, control_dt, frame)

    def to_payload(self) -> dict[str, Any]:
        """Plain tensors/metadata safe to include in a policy checkpoint."""
        return {
            "format_version": _FORMAT_VERSION,
            "centers": self.centers.detach().float().cpu().clone(),
            "scale": self.scale.detach().float().cpu().clone(),
            "control_dt": self.control_dt,
            "frame": self.frame,
            "rotation_representation": "rotvec",
            "translation_unit": "metre",
            "rotation_unit": "radian",
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> WholeActionCodebook:
        if not isinstance(payload, dict) or payload.get("format_version") != _FORMAT_VERSION:
            raise ValueError("unsupported or missing codebook format_version")
        for key, expected in (("rotation_representation", "rotvec"),
                              ("translation_unit", "metre"), ("rotation_unit", "radian")):
            if payload.get(key) != expected:
                raise ValueError(f"unsupported codebook {key}")
        required = {"centers", "scale", "control_dt", "frame"}
        if not required.issubset(payload):
            raise ValueError(f"missing codebook fields: {sorted(required - payload.keys())}")
        return cls(payload["centers"], payload["scale"], payload["control_dt"], payload["frame"])

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), destination)

    @classmethod
    def load(cls, path: str | Path) -> WholeActionCodebook:
        return cls.from_payload(torch.load(path, map_location="cpu", weights_only=True))

    def fingerprint(self) -> str:
        """Stable SHA-256 of code indices, scale, timing, and pose convention."""
        payload = self.to_payload()
        digest = hashlib.sha256()
        metadata = {key: value for key, value in payload.items() if not isinstance(value, Tensor)}
        metadata["size"] = self.size
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        for key in ("centers", "scale"):
            # Avoid requiring NumPy just to serialize small float32 buffers.
            digest.update(bytes(payload[key].contiguous().view(torch.uint8).reshape(-1).tolist()))
        return digest.hexdigest()

    @torch.no_grad()
    def reconstruction_metrics(self, actions: Tensor) -> dict[str, float | int]:
        """Physical quantization error and observed code usage on a split.

        Translation/rotvec RMSE are per coordinate. Norm P95 and SO(3) metrics
        measure per-action error. Usage reflects THIS evaluation split only;
        an unused code here is not proof that it was unused during fitting.
        """
        _check_actions(actions)
        actual = actions.to(device=self.centers.device, dtype=torch.float32).reshape(-1, _DIM)
        ids = self.encode(actual)
        reconstructed = self.decode(ids).float()
        error = actual - reconstructed
        translation_norm = error[:, :3].norm(dim=-1)
        rotation_norm = error[:, 3:].norm(dim=-1)
        geodesic = _rotation_angle_error(actual[:, 3:], reconstructed[:, 3:])
        counts = torch.bincount(ids, minlength=self.size)
        used = int((counts > 0).sum().item())
        probabilities = counts.float() / counts.sum()
        entropy = -(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()
        return {
            "translation_rmse_m": float(error[:, :3].square().mean().sqrt().item()),
            "translation_p95_m": float(torch.quantile(translation_norm, 0.95).item()),
            "rotation_rotvec_rmse_rad": float(error[:, 3:].square().mean().sqrt().item()),
            "rotation_rotvec_p95_rad": float(torch.quantile(rotation_norm, 0.95).item()),
            "rotation_geodesic_rmse_rad": float(geodesic.square().mean().sqrt().item()),
            "rotation_geodesic_p95_rad": float(torch.quantile(geodesic, 0.95).item()),
            "used_codes": used,
            "dead_codes": self.size - used,
            "usage_fraction": used / self.size,
            "perplexity": float(entropy.exp().item()),
        }
