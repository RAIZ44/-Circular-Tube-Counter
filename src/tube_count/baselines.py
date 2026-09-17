from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tube_count.config import Config
from tube_count.dataset import TubeDataset
from tube_count.metrics import count_metrics, detection_f1
from tube_count.utils import set_seed


def _hough_count(image_rgb: np.ndarray, min_radius: int, max_radius: int) -> tuple[int, np.ndarray]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(min_radius, 8),
        param1=90,
        param2=28,
        minRadius=max(min_radius - 2, 4),
        maxRadius=max_radius + 4,
    )
    if circles is None:
        return 0, np.zeros((0, 4), dtype=np.float32)
    out = []
    for x, y, r in circles[0]:
        out.append([x - r, y - r, x + r, y + r])
    return len(out), np.asarray(out, dtype=np.float32)


def run_hough(cfg: Config, split: str = "test") -> dict:
    g = cfg.generate
    ds = TubeDataset(g.root, split, g.image_size, cfg.model.stride, augment=False, seed=cfg.seed)
    preds, gts = [], []
    pred_boxes, gt_boxes = [], []
    for i in tqdm(range(len(ds)), desc=f"hough:{split}"):
        item = ds[i]
        image = (item["image"].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        count, boxes = _hough_count(image, g.min_radius, g.max_radius)
        preds.append(count)
        gts.append(int(item["count"].item()))
        pred_boxes.append(boxes)
        gt_boxes.append(item["boxes"].numpy())
    counts = count_metrics(np.asarray(preds), np.asarray(gts))
    det = detection_f1(pred_boxes, gt_boxes, cfg.eval.iou_threshold)
    return {**counts, **det, "method": "hough_circles"}


def mean_count_baseline(cfg: Config, split: str = "test") -> dict:
    g = cfg.generate
    train = TubeDataset(g.root, "train", g.image_size, cfg.model.stride, augment=False, seed=cfg.seed)
    target = TubeDataset(g.root, split, g.image_size, cfg.model.stride, augment=False, seed=cfg.seed)
    mean_count = float(np.mean([int(train[i]["count"].item()) for i in range(len(train))]))
    pred_int = int(round(mean_count))
    preds = np.full(len(target), pred_int, dtype=np.float64)
    gts = np.array([int(target[i]["count"].item()) for i in range(len(target))], dtype=np.float64)
    metrics = count_metrics(preds, gts)
    metrics["method"] = "always_mean_train_count"
    metrics["train_mean_count"] = mean_count
    metrics["constant"] = pred_int
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classical / naive counting baselines")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--split", type=str, default="test")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    set_seed(cfg.seed)
    mean_m = mean_count_baseline(cfg, args.split)
    hough_m = run_hough(cfg, args.split)
    print("mean-count:", mean_m)
    print("hough:", hough_m)


if __name__ == "__main__":
    main()
