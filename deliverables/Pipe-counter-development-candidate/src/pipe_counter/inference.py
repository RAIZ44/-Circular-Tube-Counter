"""Explicit inference precision, including compatibility for legacy checkpoints."""
from __future__ import annotations

import torch

from .utils import autocast_enabled

PRECISIONS = ('float32', 'amp', 'legacy_amp')


def checkpoint_precision(checkpoint: dict) -> str:
    precision = checkpoint.get('config', {}).get('inference_precision', 'legacy_amp')
    if precision not in PRECISIONS:
        raise ValueError(f'Unknown inference precision: {precision}')
    return precision


def inference_logits(model: torch.nn.Module, images: torch.Tensor, precision: str) -> torch.Tensor:
    if precision not in PRECISIONS:
        raise ValueError(f'Unknown inference precision: {precision}')
    with torch.autocast(device_type=images.device.type, dtype=torch.float16,
                        enabled=precision != 'float32' and autocast_enabled(images.device)):
        return model(images)


def probabilities(logits: torch.Tensor, precision: str) -> torch.Tensor:
    if precision not in PRECISIONS:
        raise ValueError(f'Unknown inference precision: {precision}')
    # FP16 sigmoid can turn distinct adjacent peaks into equal-valued plateaus.
    return (logits if precision == 'legacy_amp' else logits.float()).sigmoid()


def infer_heatmaps(model: torch.nn.Module, images: torch.Tensor, precision: str) -> torch.Tensor:
    return probabilities(inference_logits(model, images, precision), precision)
