from __future__ import annotations

import argparse
import csv
import json
import hashlib
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipe_counter.data import PipeCenterDataset
from pipe_counter.engine import validate
from pipe_counter.heatmap import modified_focal_loss
from pipe_counter.model import PipeCenterNet, create_model
from pipe_counter.peaks import tune_threshold
from pipe_counter.splitting import source_key
from pipe_counter.utils import autocast_enabled, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pipe center heatmap model")
    parser.add_argument("--manifest-dir", default="data/processed_v2")
    parser.add_argument("--output-dir", default="runs/pipe_center_v2")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-checkpoint", default=None, help="Initialize weights for a new experiment; optimizer is reset")
    parser.add_argument("--selection-metric", choices=["mean_absolute_error", "exact_count_accuracy"], default="mean_absolute_error")
    parser.add_argument('--architecture', choices=['PipeCenterNet', 'ResNet18CenterNet'], default='PipeCenterNet')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--encoder-lr-factor', type=float, default=.1)
    parser.add_argument('--inference-precision', choices=['float32', 'amp', 'legacy_amp'], default='float32')
    parser.add_argument('--validation-batch-size', type=int, default=1)
    parser.add_argument('--crop-probability', type=float, default=0.0)
    parser.add_argument('--crop-min-scale', type=float, default=0.5)
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
    temporary = path.with_suffix(".tmp")
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
                "architecture": args.architecture,
                "pretrained_encoder": args.pretrained,
                "input_normalization": "RGB in [-1,1]; ImageNet normalization inside ResNet encoder when used",
                "selection_metric": args.selection_metric,
                "inference_precision": args.inference_precision,
                "inference_batch_size": args.validation_batch_size,
                "training_crop_probability": args.crop_probability,
                "training_crop_min_scale": args.crop_min_scale,
                "init_checkpoint": args.init_checkpoint,
                "postprocessing": "content_bounds_max2000_v2",
            },
            "development_data": args.development_data,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.pretrained and args.init_checkpoint:
        raise ValueError('Choose pretrained initialization OR an initial checkpoint')
    if not 0 < args.encoder_lr_factor <= 1:
        raise ValueError('Encoder learning-rate factor must be in (0,1]')
    torch.hub.set_dir(str(Path(__file__).resolve().parents[1] / '.cache' / 'torch'))
    if args.image_size % 16 != 0:
        raise ValueError("--image-size must be divisible by 16")
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    if (output_dir / "history.csv").exists() or (output_dir / "best.pt").exists():
        raise FileExistsError("Choose a new output directory; training artifacts already exist")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "train_config.json", vars(args))

    manifest_dir = Path(args.manifest_dir)
    train_dataset = PipeCenterDataset(
        manifest_dir / "train.jsonl",
        image_size=args.image_size,
        stride=PipeCenterNet.output_stride,
        augment=True,
        crop_probability=args.crop_probability,
        crop_min_scale=args.crop_min_scale,
    )
    val_dataset = PipeCenterDataset(
        manifest_dir / "val.jsonl",
        image_size=args.image_size,
        stride=PipeCenterNet.output_stride,
        augment=False,
    )
    development_records = train_dataset.records + val_dataset.records
    args.development_data = {
        "sha256": sorted({record["sha256"] for record in development_records}),
        "source_keys": sorted({key for record in development_records
                               for key in record.get("source_keys", [source_key(record["source_file_name"])])}),
        "perceptual_group": sorted({record["perceptual_group"] for record in development_records
                                    if record.get("perceptual_group")}),
        "note": "Includes training and validation because both influence model selection.",
    }
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
        batch_size=args.validation_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = create_model(args.architecture, pretrained=args.pretrained).to(device)
    if args.init_checkpoint:
        parent = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if parent.get('config', {}).get('architecture', 'PipeCenterNet') != args.architecture:
            raise ValueError('Initial checkpoint architecture does not match the requested model')
        prior_data = parent.get("development_data")
        if not prior_data or not set(prior_data["sha256"]).issubset(set(args.development_data["sha256"])):
            raise ValueError("Initial checkpoint has unknown or out-of-split development data")
        model.load_state_dict(parent["model"])
        save_json(output_dir / "initialization.json", {
            "checkpoint": str(Path(args.init_checkpoint).resolve()),
            "sha256": hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest(),
            "parent_epoch": parent["epoch"], "optimizer_reset": True,
        })
    parameters = model.parameters()
    if args.architecture == 'ResNet18CenterNet':
        parameters = [
            {'params': model.backbone.parameters(), 'lr': args.learning_rate * args.encoder_lr_factor},
            {'params': [p for name,p in model.named_parameters() if not name.startswith('backbone.')], 'lr': args.learning_rate},
        ]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate,
                                  weight_decay=args.weight_decay, foreach=False)
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
            val_loss, peak_values, true_counts = validate(model, val_loader, device, precision=args.inference_precision)
            metrics = tune_threshold(peak_values, true_counts,
                                     thresholds=np.arange(.10, .901, .005).tolist(),
                                     objective=args.selection_metric)
            threshold = float(metrics["threshold"])
            scheduler.step(-float(metrics["exact_count_accuracy"]) if args.selection_metric == "exact_count_accuracy"
                           else float(metrics["mean_absolute_error"]))
            learning_rate = float(optimizer.param_groups[-1]["lr"])

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
            if args.selection_metric == "exact_count_accuracy":
                score = (-float(metrics["exact_count_accuracy"]), float(metrics["mean_absolute_error"]), val_loss)
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
