from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tube_count.model import TubeNet
from tube_count.targets import decode_heatmap


@torch.no_grad()
def infer_loader(
    model: TubeNet,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    stride: int,
    conf_threshold: float = 0.05,
    progress: bool = False,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    iterator = tqdm(loader, desc="infer", disable=not progress)
    for batch in iterator:
        images = batch["image"].to(device)
        heatmaps, radii = model(images)
        for i in range(images.size(0)):
            boxes, scores = decode_heatmap(
                heatmaps[i],
                radii[i],
                image_size=image_size,
                stride=stride,
                conf_threshold=conf_threshold,
            )
            gt = batch["boxes"][i].cpu().numpy()
            rows.append(
                {
                    "path": batch["path"][i],
                    "pred_boxes": boxes,
                    "scores": scores,
                    "gt_boxes": gt,
                    "gt_count": int(batch["count"][i].item()),
                    "image": (batch["image"][i].cpu().numpy().transpose(1, 2, 0) * 255.0)
                    .clip(0, 255)
                    .astype(np.uint8),
                }
            )
    return rows


def apply_threshold(rows: list[dict[str, Any]], conf_threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        keep = row["scores"] >= conf_threshold
        boxes = row["pred_boxes"][keep]
        scores = row["scores"][keep]
        item = dict(row)
        item["pred_boxes"] = boxes
        item["scores"] = scores
        item["pred_count"] = int(len(boxes))
        item["conf_threshold"] = float(conf_threshold)
        out.append(item)
    return out


def calibrate_threshold(
    rows: list[dict[str, Any]],
    sweep: np.ndarray,
) -> tuple[float, float]:
    """Pick the confidence cutoff that minimizes count MAE on ``rows``."""
    best_thr = float(sweep[0])
    best_mae = float("inf")
    gts = np.array([r["gt_count"] for r in rows], dtype=np.float64)
    for thr in sweep:
        preds = np.array(
            [int(np.sum(r["scores"] >= thr)) for r in rows],
            dtype=np.float64,
        )
        mae = float(np.mean(np.abs(preds - gts))) if len(gts) else 0.0
        if mae < best_mae:
            best_mae = mae
            best_thr = float(thr)
    return best_thr, best_mae
