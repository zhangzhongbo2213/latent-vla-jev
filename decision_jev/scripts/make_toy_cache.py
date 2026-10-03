#!/usr/bin/env python3
"""Create a deterministic, boundary-safe toy feature cache for smoke tests."""
from pathlib import Path
import argparse
import torch

from jev.data import FEATURE_CONTRACT


def make_cache(path: str, episodes: int = 32, vision_tokens: int = 4,
               vision_dim: int = 8, text_dim: int = 6, state_dim: int = 7,
               steps: int = 8, seed: int = 7, success: bool = True,
               episode_prefix: str = "episode") -> None:
    if episodes < 2 or steps < 2:
        raise ValueError("episodes and steps must be >= 2")
    generator = torch.Generator().manual_seed(seed)
    entries = []
    for index in range(episodes):
        length = steps + (index % 3)
        action = torch.randn(length, 6, generator=generator) * 0.02
        # Ensure small codebooks see distinct, nonzero actions.
        action[:, 0] += (index + 1) * 0.001
        ok = bool(success and index % 5 != 0)
        entries.append({
            "episode_id": f"{episode_prefix}-{index:04d}", "subtask_id": "subtask-0", "success": ok,
            "current_features": torch.randn(length + 1, vision_tokens, vision_dim, generator=generator),
            "terminal_features": torch.randn(vision_tokens, vision_dim, generator=generator),
            "task_features": torch.randn(3, text_dim, generator=generator),
            "task_mask": torch.ones(3, dtype=torch.bool),
            "proprio": torch.randn(length + 1, state_dim, generator=generator),
            "actions": action,
            "gripper_targets": torch.randint(0, 3, (length,), generator=generator),
        })
    payload = {"metadata": {
        "feature_contract": FEATURE_CONTRACT, "vision_encoder": "toy-vision-v1",
        "text_encoder": "toy-text-v1", "control_dt": 0.1, "action_frame": "body",
        "terminal_source": "predicted", "lwm_checkpoint": "toy-lwm-v1",
    }, "episodes": entries}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode-prefix", default="episode")
    args = parser.parse_args()
    make_cache(args.output, episodes=args.episodes, seed=args.seed,
               episode_prefix=args.episode_prefix)
