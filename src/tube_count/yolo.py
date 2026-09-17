"""YOLO-format label I/O. This is the on-disk data contract.

Each image `data/images/<split>/<stem>.png` has a sibling label file
`data/labels/<split>/<stem>.txt`. Lines are:

    <class_id> <xc> <yc> <w> <h>

All geometry is normalized to [0, 1] relative to image width/height.
Class 0 is `tube`. Empty files mean count = 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CLASS_ID = 0
CLASS_NAME = "tube"


def boxes_to_yolo(boxes_xyxy: np.ndarray, width: int, height: int) -> str:
    lines: list[str] = []
    for x1, y1, x2, y2 in boxes_xyxy.reshape(-1, 4):
        bw = max(x2 - x1, 0.0)
        bh = max(y2 - y1, 0.0)
        xc = (x1 + x2) / 2.0 / width
        yc = (y1 + y2) / 2.0 / height
        lines.append(
            f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw / width:.6f} {bh / height:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def load_yolo_boxes(path: str | Path, width: int, height: int) -> np.ndarray:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes: list[list[float]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Bad YOLO line in {path}: {raw!r}")
        _, xc, yc, w, h = (float(p) for p in parts)
        bw = w * width
        bh = h * height
        cx = xc * width
        cy = yc * height
        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def write_yolo_label(path: str | Path, boxes_xyxy: np.ndarray, width: int, height: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(boxes_to_yolo(boxes_xyxy, width, height))
