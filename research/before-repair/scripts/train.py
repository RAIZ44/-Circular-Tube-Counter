from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipe_counter.data import PipeCenterDataset
from pipe_counter.engine import validate
from pipe_counter.heatmap import modified_focal_loss
from pipe_counter.model import PipeCenterNet
from pipe_counter.peaks import tune_threshold
from pipe_counter.utils import autocast_enabled, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pipe center heatmap model")
    parser.add_argument("--manifest-dir", default="data/processed")
    parser.add_argument("--output-dir", default="runs/pipe_center_v1")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    model: PipeCenterNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    threshold: float,
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "threshold": threshold,
            "metrics": metrics,
            "config": {
                "image_size": args.image_size,
                "stride": model.output_stride,
                "architecture": "PipeCenterNet",
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.image_size % 16 != 0:
        raise ValueError("--image-size must be divisible by 16")
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "train_config.json", vars(args))

    manifest_dir = Path(args.manifest_dir)
    train_dataset = PipeCenterDataset(
        manifest_dir / "train.jsonl",
        image_size=args.image_size,
        stride=PipeCenterNet.output_stride,
        augment=True,
    )
    val_dataset = PipeCenterDataset(
        manifest_dir / "val.jsonl",
        image_size=args.image_size,
        stride=PipeCenterNet.output_stride,
        augment=False,
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = PipeCenterNet().to(device)
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    amp_enabled = autocast_enabled(device)
    # Intel client GPUs do not need GradScaler here. PyTorch's XPU guidance
    # recommends disabling it on Arc hardware that lacks native FP64 support.
    grad_scaler_enabled = device.type == "cuda"
    if grad_scaler_enabled:
        if hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda")
        else:  # Compatibility with older supported PyTorch releases.
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    print(
        f"Device: {device} | train={len(train_dataset)} | val={len(val_dataset)} "
        f"| AMP={amp_enabled} | GradScaler={grad_scaler_enabled}"
    )

    history_path = output_dir / "history.csv"
    best_score = (float("inf"), float("inf"), float("inf"))
    epochs_without_improvement = 0
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "threshold",
                "exact_count_accuracy",
                "within_one_accuracy",
                "mean_absolute_error",
                "mean_signed_error",
                "learning_rate",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_images = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
            for batch in progress:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["heatmap"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = model(images)
                # Keep the log/power-heavy focal loss in FP32 for numerical stability.
                loss = modified_focal_loss(logits.float(), targets)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                batch_size = images.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_images += batch_size
                progress.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = total_loss / max(1, total_images)
            val_loss, peak_values, true_counts = validate(model, val_loader, device)
            metrics = tune_threshold(peak_values, true_counts)
            threshold = float(metrics["threshold"])
            scheduler.step(float(metrics["mean_absolute_error"]))
            learning_rate = float(optimizer.param_groups[0]["lr"])

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "threshold": threshold,
                "exact_count_accuracy": metrics["exact_count_accuracy"],
                "within_one_accuracy": metrics["within_one_accuracy"],
                "mean_absolute_error": metrics["mean_absolute_error"],
                "mean_signed_error": metrics["mean_signed_error"],
                "learning_rate": learning_rate,
            }
            writer.writerow(row)
            history_file.flush()
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                epoch,
                threshold,
                metrics,
                args,
            )

            score = (
                float(metrics["mean_absolute_error"]),
                -float(metrics["exact_count_accuracy"]),
                val_loss,
            )
            improved = score < best_score
            if improved:
                best_score = score
                epochs_without_improvement = 0
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    threshold,
                    metrics,
                    args,
                )
                save_json(output_dir / "best_validation_metrics.json", metrics)
            else:
                epochs_without_improvement += 1

            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": round(train_loss, 5),
                        "val_loss": round(val_loss, 5),
                        "threshold": round(threshold, 3),
                        "exact_accuracy": round(
                            float(metrics["exact_count_accuracy"]), 4
                        ),
                        "mae": round(float(metrics["mean_absolute_error"]), 4),
                        "best": improved,
                    }
                )
            )
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    print(f"Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
