from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from tube_count.targets import render_targets
from tube_count.yolo import load_yolo_boxes


class TubeDataset(Dataset):
    """Reads the YOLO data contract under ``root/images/<split>`` + ``root/labels/<split>``."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int,
        stride: int,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.stride = stride
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        img_dir = self.root / "images" / split
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing image split: {img_dir}")
        self.paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not self.paths:
            raise FileNotFoundError(f"No images in {img_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def label_path(self, image_path: Path) -> Path:
        return self.root / "labels" / self.split / f"{image_path.stem}.txt"

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        with Image.open(path) as im:
            image = np.asarray(im.convert("RGB"))
        h, w = image.shape[:2]
        boxes = load_yolo_boxes(self.label_path(path), w, h)
        if (h, w) != (self.image_size, self.image_size):
            scale_x = self.image_size / w
            scale_y = self.image_size / h
            image = np.asarray(
                Image.fromarray(image).resize((self.image_size, self.image_size), Image.BILINEAR)
            )
            if boxes.size:
                boxes = boxes.copy()
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
        if self.augment:
            image, boxes = _augment(image, boxes, self.rng)
        heatmap, radius, mask = render_targets(boxes, self.image_size, self.stride)
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        return {
            "image": tensor,
            "heatmap": torch.from_numpy(heatmap).unsqueeze(0),
            "radius": torch.from_numpy(radius).unsqueeze(0),
            "radius_mask": torch.from_numpy(mask).unsqueeze(0),
            "boxes": torch.from_numpy(boxes.astype(np.float32)),
            "count": torch.tensor(len(boxes), dtype=torch.int32),
            "path": str(path),
        }


def _augment(image: np.ndarray, boxes: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if float(rng.random()) < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        w = image.shape[1]
        if boxes.size:
            flipped = boxes.copy()
            flipped[:, 0] = w - boxes[:, 2]
            flipped[:, 2] = w - boxes[:, 0]
            boxes = flipped
    if float(rng.random()) < 0.8:
        gain = float(rng.uniform(0.75, 1.25))
        bias = float(rng.uniform(-12, 12))
        image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    return image, boxes


def collate_tubes(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "heatmap": torch.stack([b["heatmap"] for b in batch]),
        "radius": torch.stack([b["radius"] for b in batch]),
        "radius_mask": torch.stack([b["radius_mask"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "count": torch.stack([b["count"] for b in batch]),
        "path": [b["path"] for b in batch],
    }


def make_loader(
    root: str | Path,
    split: str,
    image_size: int,
    stride: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TubeDataset(root, split, image_size, stride, augment=augment, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_tubes,
        drop_last=False,
        pin_memory=False,
    )
