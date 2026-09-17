from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def overlay_boxes(
    image: np.ndarray,
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    gt_count: int | None,
    pred_count: int,
    path: str | Path | None = None,
) -> Image.Image:
    canvas = Image.fromarray(image.copy()).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for box in np.asarray(gt_boxes).reshape(-1, 4) if np.asarray(gt_boxes).size else []:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        draw.ellipse([x1, y1, x2, y2], outline=(0, 200, 70), width=2)
    for box in np.asarray(pred_boxes).reshape(-1, 4) if np.asarray(pred_boxes).size else []:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(230, 40, 40), width=2)
    if gt_count is None:
        label = f"PRED={pred_count}"
    else:
        label = f"GT={gt_count}  PRED={pred_count}"
    draw.rectangle([2, 2, 8 + 8 * len(label), 22], fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255), font=_font())
    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out)
    return canvas
