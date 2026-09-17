from __future__ import annotations

import numpy as np
import torch

from tube_count.config import GenerateConfig
from tube_count.generate import render_scene
from tube_count.model import TubeNet, detection_loss
from tube_count.yolo import boxes_to_yolo, load_yolo_boxes


def test_render_scene_labels_in_range() -> None:
    rng = np.random.default_rng(0)
    cfg = GenerateConfig(image_size=128, min_count=1, max_count=5, min_radius=6, max_radius=16)
    img, boxes = render_scene(cfg, rng)
    assert img.shape == (128, 128, 3)
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert 1 <= len(boxes) <= 5
    text = boxes_to_yolo(boxes, 128, 128)
    assert text.count("\n") == len(boxes) or (len(boxes) == 0 and text == "")
    for line in text.strip().splitlines():
        cls, xc, yc, w, h = line.split()
        assert cls == "0"
        for v in (xc, yc, w, h):
            assert 0.0 <= float(v) <= 1.0


def test_yolo_roundtrip(tmp_path) -> None:
    boxes = np.array([[10.0, 20.0, 50.0, 60.0]], dtype=np.float32)
    path = tmp_path / "a.txt"
    path.write_text(boxes_to_yolo(boxes, 100, 80))
    got = load_yolo_boxes(path, 100, 80)
    np.testing.assert_allclose(got, boxes, atol=1e-3)


def test_tubenet_forward_and_loss() -> None:
    model = TubeNet(base_channels=8, stride=4)
    x = torch.zeros(2, 3, 64, 64)
    hm, rad = model(x)
    assert hm.shape == (2, 1, 16, 16)
    assert rad.shape == (2, 1, 16, 16)
    loss, parts = detection_loss(hm, rad, torch.zeros_like(hm), torch.zeros_like(rad), torch.zeros_like(rad))
    assert torch.isfinite(loss)
    assert "hm_loss" in parts
