"""Validated feature caches and boundary-safe subtask training samples."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

FEATURE_CONTRACT = "terminal_conditioned_jev_v1"


@dataclass
class FeatureCache:
    metadata: dict[str, Any]
    episodes: list[dict[str, Any]]

    @property
    def dimensions(self) -> dict[str, int]:
        episode = self.episodes[0]
        return {
            "vision_dim": episode["current_features"].shape[-1],
            "text_dim": episode["task_features"].shape[-1],
            "state_dim": episode["proprio"].shape[-1],
        }

    @property
    def source_ids(self) -> list[list[str]]:
        return [[str(e["episode_id"]), str(e["subtask_id"])] for e in self.episodes]

    @property
    def successful_source_ids(self) -> list[list[str]]:
        return [[str(e["episode_id"]), str(e["subtask_id"])] for e in self.episodes if e["success"]]

    def actions(self) -> torch.Tensor:
        actions = [e["actions"] for e in self.episodes if e["success"]]
        if not actions:
            raise ValueError("Codebook fitting requires successful training demonstrations")
        return torch.cat(actions, dim=0)


def _tensor(value: Any, name: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a rank-{ndim} torch.Tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite floating point values")
    return value.detach().cpu().float()


def load_cache(path: str | Path) -> FeatureCache:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return validate_cache(payload)


def load_feature_cache(path: str | Path) -> FeatureCache:
    """Explicit alias used by training/controller integrations."""
    return load_cache(path)


def save_cache(path: str | Path, cache: FeatureCache) -> None:
    """Persist a validated cache without changing tensor contents."""
    validated = validate_cache({"metadata": cache.metadata, "episodes": cache.episodes})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": validated.metadata, "episodes": validated.episodes}, path)


def validate_cache(payload: dict[str, Any]) -> FeatureCache:
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("Cache must contain metadata and episodes")
    metadata = dict(payload["metadata"])
    for key in ("feature_contract", "vision_encoder", "text_encoder", "control_dt", "action_frame", "terminal_source"):
        if key not in metadata:
            raise ValueError(f"Cache metadata is missing {key}")
    if metadata["feature_contract"] != FEATURE_CONTRACT:
        raise ValueError(f"Unsupported feature_contract: {metadata['feature_contract']}")
    if not metadata["vision_encoder"] or not metadata["text_encoder"]:
        raise ValueError("Encoder identities must be nonempty")
    if not isinstance(metadata["control_dt"], (int, float)) or not 0 < metadata["control_dt"] < float("inf"):
        raise ValueError("control_dt must be finite and positive")
    if metadata["action_frame"] not in ("body", "world"):
        raise ValueError("action_frame must be body or world")
    if metadata["terminal_source"] not in ("predicted", "oracle"):
        raise ValueError("terminal_source must be predicted or oracle")
    if metadata["terminal_source"] == "predicted" and not metadata.get("lwm_checkpoint"):
        raise ValueError("Predicted terminal features require lwm_checkpoint provenance")
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("Cache episodes must be a nonempty list")
    episodes, identities, dimensions = [], set(), None
    visual_token_count = None
    for raw in raw_episodes:
        if not isinstance(raw, dict):
            raise ValueError("Each episode/subtask entry must be a dictionary")
        for key in ("episode_id", "subtask_id", "success"):
            if key not in raw:
                raise ValueError(f"Subtask entry is missing {key}")
        identity = (str(raw["episode_id"]), str(raw["subtask_id"]))
        if identity in identities:
            raise ValueError(f"Duplicate episode/subtask identity: {identity}")
        identities.add(identity)
        if not isinstance(raw["success"], bool):
            raise ValueError("success must be boolean")
        e = dict(raw)
        for key, ndim in (("current_features", 3), ("terminal_features", 2), ("task_features", 2), ("proprio", 2), ("actions", 2)):
            e[key] = _tensor(raw.get(key), key, ndim)
        length = e["actions"].shape[0]
        if length < 1 or e["actions"].shape[-1] != 6:
            raise ValueError("actions must have shape [T, 6] with T >= 1")
        if e["current_features"].shape[0] != length + 1 or e["proprio"].shape[0] != length + 1:
            raise ValueError("Observations and proprio must include the terminal state: T+1 frames")
        if e["terminal_features"].shape != e["current_features"].shape[1:]:
            raise ValueError("terminal_features must match current_features spatial/feature dimensions")
        if min(e["current_features"].shape[1:]) < 1 or min(e["task_features"].shape) < 1 or e["proprio"].shape[-1] < 1:
            raise ValueError("Feature and state dimensions must be positive")
        mask = raw.get("task_mask")
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (e["task_features"].shape[0],) or not bool(mask.any()):
            raise ValueError("task_mask must be boolean [L], with at least one valid token")
        e["task_mask"] = mask.cpu()
        gripper = raw.get("gripper_targets")
        if gripper is not None:
            if not isinstance(gripper, torch.Tensor) or gripper.shape != (length,) or gripper.dtype not in (torch.int64, torch.int32):
                raise ValueError("gripper_targets must be integer [T]")
            if bool(((gripper < 0) | (gripper > 2)).any()):
                raise ValueError("Gripper classes are 0=hold, 1=open, 2=close")
            e["gripper_targets"] = gripper.cpu().long()
        token_count = e["current_features"].shape[1]
        dims = (e["current_features"].shape[-1], e["task_features"].shape[-1], e["proprio"].shape[-1])
        if dimensions is None:
            dimensions = dims
            visual_token_count = token_count
        elif dims != dimensions:
            raise ValueError("Feature dimensions differ within cache")
        elif token_count != visual_token_count:
            raise ValueError("Visual token counts differ within cache; batch them separately")
        episodes.append(e)
    return FeatureCache(metadata, episodes)


def validate_pair(train: FeatureCache, validation: FeatureCache) -> None:
    overlap = {str(e["episode_id"]) for e in train.episodes} & {str(e["episode_id"]) for e in validation.episodes}
    if overlap:
        raise ValueError(f"Train/validation episode leakage: {sorted(overlap)}")
    if train.dimensions != validation.dimensions:
        raise ValueError("Train/validation feature dimensions differ")
    for key in ("feature_contract", "vision_encoder", "text_encoder", "control_dt", "action_frame", "terminal_source", "lwm_checkpoint"):
        if train.metadata.get(key) != validation.metadata.get(key):
            raise ValueError(f"Train/validation metadata differs: {key}")


class SubtaskDataset(Dataset):
    """One sample per state, including the no-action terminal state."""

    def __init__(self, cache: FeatureCache, horizon: int = 4, train: bool = False, progress_noise: float = 0.0):
        if horizon < 1 or progress_noise < 0:
            raise ValueError("horizon must be positive and progress_noise nonnegative")
        self.cache, self.horizon, self.train, self.progress_noise = cache, horizon, train, progress_noise
        self.indices = [(i, t) for i, e in enumerate(cache.episodes) for t in range(e["actions"].shape[0] + 1)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        i, t = self.indices[index]
        e = self.cache.episodes[i]
        total, horizon = e["actions"].shape[0], self.horizon
        remaining = min(horizon, total - t)
        action = torch.zeros(horizon, 6)
        mask = torch.zeros(horizon, dtype=torch.bool)
        action[:remaining] = e["actions"][t:t + remaining]
        mask[:remaining] = e["success"]
        progress = t / total if e["success"] else 0.0
        if self.train and self.progress_noise and t > 0:
            progress = float((torch.tensor(progress) + torch.randn(()) * self.progress_noise).clamp(0, 1))
        targets = torch.tensor([min(t + h + 1, total) / total for h in range(horizon)])
        gripper = torch.zeros(horizon, dtype=torch.long)
        gripper_mask = torch.zeros(horizon, dtype=torch.bool)
        if "gripper_targets" in e:
            gripper[:remaining] = e["gripper_targets"][t:t + remaining]
            gripper_mask[:remaining] = e["success"]
        return {
            "current_features": e["current_features"][t], "terminal_features": e["terminal_features"],
            "task_features": e["task_features"], "task_mask": e["task_mask"], "proprio": e["proprio"][t],
            "progress": torch.tensor(progress), "actions": action, "action_mask": mask,
            "progress_targets": targets, "progress_mask": mask.clone(),
            "completion": torch.tensor(float(e["success"] and t == total)),
            "gripper_targets": gripper, "gripper_mask": gripper_mask,
            "episode_id": str(e["episode_id"]), "subtask_id": str(e["subtask_id"]), "step": t,
        }


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad observation/text token axes; return True for valid tokens."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    visual_tokens = max(s["current_features"].shape[0] for s in samples)
    if any(s["current_features"].shape[0] != visual_tokens for s in samples):
        raise ValueError("Variable visual token counts require separate batches (no visual mask in model API)")
    text_length = max(s["task_features"].shape[0] for s in samples)
    task = samples[0]["task_features"].new_zeros(len(samples), text_length, samples[0]["task_features"].shape[-1])
    task_mask = torch.zeros(len(samples), text_length, dtype=torch.bool)
    for i, sample in enumerate(samples):
        n = sample["task_features"].shape[0]
        task[i, :n], task_mask[i, :n] = sample["task_features"], sample["task_mask"]
    batch = {"task_features": task, "task_mask": task_mask}
    for key in samples[0]:
        if key in batch:
            continue
        values = [s[key] for s in samples]
        batch[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return batch
