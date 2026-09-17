from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.nn import functional as F


def local_maximum_mask(heatmaps: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be a positive odd number")
    pooled = F.max_pool2d(
        heatmaps, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
    )
    return heatmaps.eq(pooled)


def peak_confidences(
    heatmaps: torch.Tensor, kernel_size: int = 3
) -> list[np.ndarray]:
    """Return confidence values for all positive local maxima in each image."""
    mask = local_maximum_mask(heatmaps, kernel_size)
    outputs: list[np.ndarray] = []
    for heatmap, image_mask in zip(heatmaps, mask, strict=True):
        values = heatmap[image_mask & heatmap.gt(0)].detach().cpu().numpy()
        outputs.append(np.asarray(values, dtype=np.float32))
    return outputs


def extract_peaks(
    heatmap: torch.Tensor,
    threshold: float,
    kernel_size: int = 3,
    max_peaks: int = 2000,
) -> list[tuple[float, float, float]]:
    """Extract (x, y, confidence) peaks from a 2D heatmap."""
    if heatmap.ndim != 2:
        raise ValueError("extract_peaks expects a 2D heatmap")
    batched = heatmap[None, None, ...]
    mask = local_maximum_mask(batched, kernel_size)[0, 0]
    ys, xs = torch.where(mask & heatmap.ge(threshold))
    if len(xs) == 0:
        return []
    scores = heatmap[ys, xs]
    order = torch.argsort(scores, descending=True)[:max_peaks]
    return [
        (float(xs[i].item()), float(ys[i].item()), float(scores[i].item()))
        for i in order
    ]


def count_metrics(
    peak_values: Sequence[np.ndarray],
    true_counts: Sequence[int],
    threshold: float,
) -> dict[str, float | int]:
    predictions = np.asarray(
        [int(np.count_nonzero(values >= threshold)) for values in peak_values],
        dtype=np.int64,
    )
    truth = np.asarray(true_counts, dtype=np.int64)
    if len(truth) == 0:
        raise ValueError("No samples were supplied for metrics")
    absolute_error = np.abs(predictions - truth)
    return {
        "images": int(len(truth)),
        "threshold": float(threshold),
        "exact_count_accuracy": float(np.mean(absolute_error == 0)),
        "within_one_accuracy": float(np.mean(absolute_error <= 1)),
        "mean_absolute_error": float(np.mean(absolute_error)),
        "root_mean_squared_error": float(
            np.sqrt(np.mean((predictions - truth).astype(np.float64) ** 2))
        ),
        "mean_signed_error": float(np.mean(predictions - truth)),
        "total_true": int(truth.sum()),
        "total_predicted": int(predictions.sum()),
    }


def tune_threshold(
    peak_values: Sequence[np.ndarray],
    true_counts: Sequence[int],
    thresholds: Sequence[float] | None = None,
) -> dict[str, float | int]:
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.025).tolist()
    candidates = [count_metrics(peak_values, true_counts, value) for value in thresholds]
    return min(
        candidates,
        key=lambda item: (
            float(item["mean_absolute_error"]),
            -float(item["exact_count_accuracy"]),
            abs(float(item["mean_signed_error"])),
        ),
    )

