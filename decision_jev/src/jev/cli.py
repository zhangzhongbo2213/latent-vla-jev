"""Offline cache utilities and lightweight training entry points."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import torch
from torch.utils.data import DataLoader

from .data import load_cache, validate_pair, SubtaskDataset, collate_batch
from .codebook import WholeActionCodebook
from .model import DecisionJEV, JEVConfig
from .checkpoint import save_checkpoint


def _loader(cache, horizon, batch_size, train=False, progress_noise=0.0):
    return DataLoader(SubtaskDataset(cache, horizon=horizon, train=train, progress_noise=progress_noise), batch_size=batch_size,
                      shuffle=train, collate_fn=collate_batch)


def fit_codebook(args):
    cache = load_cache(args.cache)
    actions = cache.actions()
    codebook = WholeActionCodebook.fit(actions, size=args.size, iterations=args.iterations,
                                       seed=args.seed, control_dt=cache.metadata["control_dt"],
                                       frame=cache.metadata["action_frame"])
    codebook.save(args.output)
    print(json.dumps({"size": codebook.size, "fingerprint": codebook.fingerprint(),
                      "fit_source_ids": cache.successful_source_ids}))


def evaluate(args):
    cache = load_cache(args.cache)
    codebook = WholeActionCodebook.load(args.codebook)
    print(json.dumps(codebook.reconstruction_metrics(cache.actions()), indent=2))


def train(args):
    torch.manual_seed(args.seed)
    train_cache = load_cache(args.train)
    val_cache = load_cache(args.val)
    validate_pair(train_cache, val_cache)
    codebook = WholeActionCodebook.load(args.codebook) if args.codebook else WholeActionCodebook.fit(
        train_cache.actions(), size=args.codebook_size, seed=args.seed,
        control_dt=train_cache.metadata["control_dt"], frame=train_cache.metadata["action_frame"])
    if codebook.size != args.codebook_size and args.codebook:
        raise ValueError("External codebook size does not match --codebook-size")
    if codebook.frame != train_cache.metadata["action_frame"] or not math.isclose(
            codebook.control_dt, float(train_cache.metadata["control_dt"]), rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("Codebook control_dt/frame do not match the feature cache")
    dims = train_cache.dimensions
    config = JEVConfig(**dims, codebook_size=codebook.size, horizon=args.horizon,
                       hidden_dim=args.hidden_dim, heads=args.heads,
                       gripper_classes=3 if any("gripper_targets" in e for e in train_cache.episodes) else 0)
    model = DecisionJEV(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Keep this command dependency-light; loss implementations can be supplied by package root.
    try:
        from .losses import jev_loss
    except ImportError as exc:
        raise RuntimeError("jev.losses is required for train") from exc
    loader = _loader(train_cache, args.horizon, args.batch_size, train=True)
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            outputs = model(batch["current_features"], batch["terminal_features"], batch["task_features"],
                            batch["proprio"], batch["progress"], batch["task_mask"])
            result = jev_loss(outputs, batch, codebook)
            loss = result["loss"] if isinstance(result, dict) else result
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    save_checkpoint(args.output, model, codebook,
                    {"train_cache": str(Path(args.train)), "val_cache": str(Path(args.val)),
                     "fit_source_ids": train_cache.successful_source_ids}, optimizer=optimizer,
                    epoch=args.epochs)
    print(f"saved {args.output}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jev")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-codebook"); fit.add_argument("cache"); fit.add_argument("output"); fit.add_argument("--size", type=int, default=1024); fit.add_argument("--iterations", type=int, default=30); fit.add_argument("--seed", type=int, default=0); fit.set_defaults(func=fit_codebook)
    ev = sub.add_parser("evaluate"); ev.add_argument("cache"); ev.add_argument("codebook"); ev.set_defaults(func=evaluate)
    tr = sub.add_parser("train"); tr.add_argument("train"); tr.add_argument("val"); tr.add_argument("output"); tr.add_argument("--codebook"); tr.add_argument("--codebook-size", type=int, default=1024); tr.add_argument("--horizon", type=int, default=4); tr.add_argument("--hidden-dim", type=int, default=256); tr.add_argument("--heads", type=int, default=8); tr.add_argument("--batch-size", type=int, default=8); tr.add_argument("--epochs", type=int, default=1); tr.add_argument("--lr", type=float, default=1e-4); tr.add_argument("--seed", type=int, default=0); tr.set_defaults(func=train)
    args = parser.parse_args(argv); args.func(args)


if __name__ == "__main__":
    main()
