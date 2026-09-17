from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .heatmap import build_heatmap
from .utils import read_jsonl


def letterbox(
    image: np.ndarray,
    boxes: list[list[float]],
    image_size: int,
) -> tuple[np.ndarray, list[list[float]], dict[str, float]]:
    original_height, original_width = image.shape[:2]
    scale = min(image_size / original_width, image_size / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    pad_x = (image_size - resized_width) // 2
    pad_y = (image_size - resized_height) // 2
    canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

    transformed = [
        [
            center_x * scale + pad_x,
            center_y * scale + pad_y,
            box_width * scale,
            box_height * scale,
        ]
        for center_x, center_y, box_width, box_height in boxes
    ]
    meta = {
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "original_width": float(original_width),
        "original_height": float(original_height),
    }
    return canvas, transformed, meta


def normalize_image(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    array = image_rgb.astype(np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array.transpose(2, 0, 1)).float()


def content_bounds(meta: dict[str, float], stride: int) -> list[float]:
    """Half-open bounds of original-image coordinates on the heatmap grid."""
    return [meta["pad_x"] / stride, meta["pad_y"] / stride,
            (meta["pad_x"] + meta["original_width"] * meta["scale"]) / stride,
            (meta["pad_y"] + meta["original_height"] * meta["scale"]) / stride]


class PipeCenterDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 512,
        stride: int = 4,
        augment: bool = False,
    ) -> None:
        self.records = read_jsonl(manifest)
        self.image_size = image_size
        self.stride = stride
        self.augment = augment
        if not self.records:
            raise RuntimeError(f"Manifest contains no records: {manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = cv2.imread(record["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read image: {record['image_path']}")
        boxes = [list(map(float, box)) for box in record["boxes"]]
        image, boxes, meta = letterbox(image, boxes, self.image_size)
        bounds = content_bounds(meta, self.stride)

        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            for box in boxes:
                box[0] = self.image_size - 1 - box[0]
            bounds[0], bounds[2] = self.image_size / self.stride - bounds[2], self.image_size / self.stride - bounds[0]

        if self.augment:
            alpha = random.uniform(0.82, 1.18)
            beta = random.uniform(-18.0, 18.0)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        heatmap = build_heatmap(boxes, self.image_size, self.stride)
        return {
            "image": normalize_image(image),
            "heatmap": torch.from_numpy(heatmap[None, ...]).float(),
            "count": torch.tensor(len(boxes), dtype=torch.long),
            "path": record["image_path"],
            "content_bounds": torch.tensor(bounds, dtype=torch.float32),
        }
