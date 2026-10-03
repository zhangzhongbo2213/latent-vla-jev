#!/usr/bin/env python3
"""Create random feature caches for checking the software path only."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def make_cache(path: Path, count: int, offset: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    vision_dim, text_dim, visual_tokens, text_tokens = 32, 24, 16, 8
    current = torch.randn(count, visual_tokens, vision_dim, generator=generator)
    task = torch.randn(count, text_tokens, text_dim, generator=generator)
    # Keep the toy mapping learnable while retaining the real tensor contract.
    task_signal = task.mean(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
    terminal = current + 0.1 * task_signal.expand(-1, visual_tokens, vision_dim)
    terminal = terminal + 0.02 * torch.randn(
        terminal.shape, generator=generator
    )
    payload = {
        "current_features": current,
        "terminal_features": terminal,
        "task_features": task,
        "task_mask": torch.ones(count, text_tokens, dtype=torch.bool),
        "success": torch.ones(count, dtype=torch.bool),
        "episode_ids": [f"episode-{offset + i:05d}" for i in range(count)],
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/toy_cache")
    parser.add_argument("--train-count", type=int, default=48)
    parser.add_argument("--val-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    make_cache(output / "train.pt", args.train_count, 0, args.seed)
    make_cache(output / "val.pt", args.val_count, args.train_count, args.seed + 1)
    print(f"Wrote software-only toy caches to {output}")


if __name__ == "__main__":
    main()
