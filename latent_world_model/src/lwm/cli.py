"""Training entry point for the terminal-feature predictor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import FeatureCacheDataset
from .losses import terminal_feature_loss
from .model import TerminalFeaturePredictor


def _evaluate(model, loader, device):
    model.eval()
    totals = {"loss": 0.0, "cosine": 0.0, "smooth_l1": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            current = batch["current_features"].to(device)
            task = batch["task_features"].to(device)
            mask = batch["task_mask"].to(device)
            target = batch["terminal_features"].to(device)
            metrics = terminal_feature_loss(model(current, task, mask), target)
            size = current.shape[0]
            count += size
            for key in totals:
                totals[key] += float(metrics[key]) * size
    return {key: value / max(count, 1) for key, value in totals.items()}


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    train_set = FeatureCacheDataset(args.train)
    val_set = FeatureCacheDataset(args.val)
    overlap = train_set.episode_ids & val_set.episode_ids
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(f"Episode leakage between train and val caches: {examples}")
    if (train_set.vision_dim, train_set.text_dim) != (val_set.vision_dim, val_set.text_dim):
        raise ValueError("Train and validation feature dimensions differ")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = TerminalFeaturePredictor(
        vision_dim=train_set.vision_dim,
        text_dim=train_set.text_dim,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        latent_count=args.latent_count,
        latent_layers=args.latent_layers,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            current = batch["current_features"].to(device)
            task = batch["task_features"].to(device)
            mask = batch["task_mask"].to(device)
            target = batch["terminal_features"].to(device)
            optimizer.zero_grad(set_to_none=True)
            metrics = terminal_feature_loss(model(current, task, mask), target)
            if not torch.isfinite(metrics["loss"]):
                raise FloatingPointError("Non-finite terminal feature loss")
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            size = current.shape[0]
            total += float(metrics["loss"].detach()) * size
            count += size
        val_metrics = _evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total / max(count, 1), **val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "model_config": {
                "vision_dim": train_set.vision_dim,
                "text_dim": train_set.text_dim,
                "hidden_dim": args.hidden_dim,
                "heads": args.heads,
                "latent_count": args.latent_count,
                "latent_layers": args.latent_layers,
                "decoder_layers": args.decoder_layers,
                "dropout": args.dropout,
            },
            "feature_contract": "frozen_terminal_image_tokens_v1",
        }
        torch.save(state, output / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(state, output / "best.pt")
    (output / "metrics.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output / "best.pt"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="train on cached feature tensors")
    train_parser.add_argument("--train", required=True, help="training .pt feature cache")
    train_parser.add_argument("--val", required=True, help="validation .pt feature cache")
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--heads", type=int, default=8)
    train_parser.add_argument("--latent-count", type=int, default=16)
    train_parser.add_argument("--latent-layers", type=int, default=4)
    train_parser.add_argument("--decoder-layers", type=int, default=2)
    train_parser.add_argument("--dropout", type=float, default=0.0)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "train":
        best = train(args)
        print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
