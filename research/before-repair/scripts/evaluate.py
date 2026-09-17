from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from pipe_counter.data import PipeCenterDataset
from pipe_counter.engine import validate
from pipe_counter.model import load_checkpoint_model
from pipe_counter.peaks import count_metrics, tune_threshold
from pipe_counter.utils import resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exact pipe count accuracy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/processed/test.jsonl")
    parser.add_argument("--output", default="runs/test_metrics.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Only use on validation data; tuning on test data leaks test information.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    config = checkpoint.get("config", {})
    dataset = PipeCenterDataset(
        args.manifest,
        image_size=int(config.get("image_size", 512)),
        stride=int(config.get("stride", 4)),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    loss, peak_values, true_counts = validate(model, loader, device)
    if args.tune_threshold:
        metrics = tune_threshold(peak_values, true_counts)
        metrics["threshold_source"] = "tuned_on_supplied_manifest"
    else:
        threshold = float(checkpoint.get("threshold", 0.3))
        metrics = count_metrics(peak_values, true_counts, threshold)
        metrics["threshold_source"] = "checkpoint_validation_threshold"
    metrics["heatmap_loss"] = loss
    metrics["checkpoint"] = str(Path(args.checkpoint).resolve())
    metrics["manifest"] = str(Path(args.manifest).resolve())
    save_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
