from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .heatmap import modified_focal_loss
from .peaks import peak_confidences
from .inference import inference_logits, probabilities


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    precision: str = 'float32',
) -> tuple[float, list, list[int]]:
    model.eval()
    total_loss = 0.0
    total_images = 0
    all_peak_values: list = []
    all_counts: list[int] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["heatmap"].to(device, non_blocking=True)
        logits = inference_logits(model, images, precision)
        loss = modified_focal_loss(logits.float(), targets)
        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_images += batch_size
        all_peak_values.extend(peak_confidences(probabilities(logits, precision), bounds=batch.get("content_bounds")))
        all_counts.extend(int(value) for value in batch["count"].tolist())
    if total_images == 0:
        raise RuntimeError("Validation loader produced no images")
    return total_loss / total_images, all_peak_values, all_counts
