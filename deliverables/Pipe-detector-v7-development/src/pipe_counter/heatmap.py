from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch


def draw_gaussian(
    heatmap: np.ndarray, center_x: float, center_y: float, sigma: float
) -> None:
    """Draw a truncated 2D Gaussian into heatmap using a pixelwise maximum."""
    sigma = max(float(sigma), 0.75)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    center_x_int = int(round(center_x))
    center_y_int = int(round(center_y))
    height, width = heatmap.shape

    left = max(0, center_x_int - radius)
    right = min(width, center_x_int + radius + 1)
    top = max(0, center_y_int - radius)
    bottom = min(height, center_y_int + radius + 1)
    if left >= right or top >= bottom:
        return

    # CenterNet targets place the peak exactly on an output-grid pixel. This
    # guarantees a target value of 1.0 for the positive focal-loss location.
    xs = np.arange(left, right, dtype=np.float32) - float(center_x_int)
    ys = np.arange(top, bottom, dtype=np.float32) - float(center_y_int)
    gaussian = np.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / (2.0 * sigma**2))
    patch = heatmap[top:bottom, left:right]
    np.maximum(patch, gaussian, out=patch)


def build_heatmap(
    boxes: Sequence[Sequence[float]], image_size: int, stride: int = 4
) -> np.ndarray:
    output_size = image_size // stride
    heatmap = np.zeros((output_size, output_size), dtype=np.float32)
    for center_x, center_y, box_width, box_height in boxes:
        x = float(center_x) / stride
        y = float(center_y) / stride
        apparent_diameter = min(float(box_width), float(box_height)) / stride
        sigma = float(np.clip(apparent_diameter / 6.0, 0.9, 6.0))
        draw_gaussian(heatmap, x, y, sigma)
    return heatmap


def modified_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CenterNet-style focal loss for a one-channel center heatmap."""
    prediction = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
    positive = target.eq(1.0).float()
    negative = target.lt(1.0).float()
    negative_weights = torch.pow(1.0 - target, 4.0)

    positive_loss = (
        torch.log(prediction) * torch.pow(1.0 - prediction, 2.0) * positive
    )
    negative_loss = (
        torch.log(1.0 - prediction)
        * torch.pow(prediction, 2.0)
        * negative_weights
        * negative
    )
    positive_count = positive.sum()
    if positive_count.item() == 0:
        return -negative_loss.sum()
    return -(positive_loss.sum() + negative_loss.sum()) / positive_count
