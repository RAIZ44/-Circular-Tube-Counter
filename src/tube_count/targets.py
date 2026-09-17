from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def gaussian2d(shape: tuple[int, int], sigma: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    kernel = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    kernel[kernel < np.finfo(np.float32).eps * kernel.max()] = 0
    return kernel.astype(np.float32)


def draw_gaussian(heatmap: np.ndarray, cx: float, cy: float, radius_px: float, stride: int) -> None:
    sigma = max(radius_px / max(stride, 1) / 2.0, 1.0)
    radius = int(max(2, math.ceil(3 * sigma)))
    diameter = 2 * radius + 1
    kernel = gaussian2d((diameter, diameter), sigma)
    hm_h, hm_w = heatmap.shape
    x, y = int(round(cx)), int(round(cy))
    left, top = x - radius, y - radius
    right, bottom = x + radius + 1, y + radius + 1
    k_l = max(0, -left)
    k_t = max(0, -top)
    k_r = diameter - max(0, right - hm_w)
    k_b = diameter - max(0, bottom - hm_h)
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, hm_w)
    bottom = min(bottom, hm_h)
    if right <= left or bottom <= top:
        return
    patch = heatmap[top:bottom, left:right]
    k = kernel[k_t:k_b, k_l:k_r]
    np.maximum(patch, k, out=patch)


def render_targets(
    boxes_xyxy: np.ndarray,
    image_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hm_size = image_size // stride
    heatmap = np.zeros((hm_size, hm_size), dtype=np.float32)
    radius_map = np.zeros((hm_size, hm_size), dtype=np.float32)
    mask = np.zeros((hm_size, hm_size), dtype=np.float32)
    if boxes_xyxy.size == 0:
        return heatmap, radius_map, mask
    for x1, y1, x2, y2 in boxes_xyxy.reshape(-1, 4):
        cx = ((x1 + x2) / 2.0) / stride
        cy = ((y1 + y2) / 2.0) / stride
        radius_px = 0.25 * ((x2 - x1) + (y2 - y1))
        draw_gaussian(heatmap, cx, cy, radius_px, stride)
        ix = int(round(cx))
        iy = int(round(cy))
        if 0 <= ix < hm_size and 0 <= iy < hm_size:
            radius_map[iy, ix] = radius_px / image_size
            mask[iy, ix] = 1.0
    return heatmap, radius_map, mask


def center_focal_loss(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CornerNet / CenterNet penalty-reduced focal loss. ``pred`` is a probability map."""
    pred = pred.clamp(1e-6, 1.0 - 1e-6)
    pos = (gt == 1).float()
    neg = (gt < 1).float()
    pos_loss = -((1.0 - pred) ** alpha) * torch.log(pred) * pos
    neg_loss = -((1.0 - gt) ** beta) * (pred**alpha) * torch.log(1.0 - pred) * neg
    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def radius_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp(min=1.0)
    return (F.l1_loss(pred, target, reduction="none") * mask).sum() / denom


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    iou = np.zeros((len(a), len(b)), dtype=np.float32)
    for i, box in enumerate(a):
        xx1 = np.maximum(box[0], b[:, 0])
        yy1 = np.maximum(box[1], b[:, 1])
        xx2 = np.minimum(box[2], b[:, 2])
        yy2 = np.minimum(box[3], b[:, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = area_a[i] + area_b - inter
        iou[i] = inter / np.clip(union, 1e-6, None)
    return iou


def decode_heatmap(
    heatmap: torch.Tensor,
    radius_map: torch.Tensor,
    image_size: int,
    stride: int,
    conf_threshold: float,
    nms_kernel: int = 3,
    max_dets: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a single (1,H,W) heatmap + radius map into xyxy boxes and scores."""
    if heatmap.ndim == 4:
        heatmap = heatmap[0]
        radius_map = radius_map[0]
    if heatmap.ndim == 3:
        heatmap = heatmap[0]
        radius_map = radius_map[0]
    hm = heatmap.unsqueeze(0).unsqueeze(0)
    pad = nms_kernel // 2
    pooled = F.max_pool2d(hm, kernel_size=nms_kernel, stride=1, padding=pad)
    peaks = ((pooled == hm) & (hm >= conf_threshold)).squeeze(0).squeeze(0)
    ys, xs = torch.where(peaks)
    scores = hm[0, 0, ys, xs]
    if scores.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if scores.numel() > max_dets:
        top = torch.topk(scores, max_dets)
        scores = top.values
        ys = ys[top.indices]
        xs = xs[top.indices]
    radii = radius_map[ys, xs].clamp(min=1e-4) * image_size
    cx = (xs.float() + 0.5) * stride
    cy = (ys.float() + 0.5) * stride
    x1 = (cx - radii).clamp(0, image_size - 1)
    y1 = (cy - radii).clamp(0, image_size - 1)
    x2 = (cx + radii).clamp(0, image_size - 1)
    y2 = (cy + radii).clamp(0, image_size - 1)
    boxes = torch.stack([x1, y1, x2, y2], dim=1).cpu().numpy().astype(np.float32)
    return boxes, scores.detach().cpu().numpy().astype(np.float32)
