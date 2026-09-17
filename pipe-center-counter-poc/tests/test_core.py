from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from pipe_counter.geometry import analyze_lattice
from pipe_counter.heatmap import build_heatmap, modified_focal_loss
from pipe_counter.model import PipeCenterNet
from pipe_counter.peaks import extract_peaks
from pipe_counter.prepare import prepare_dataset


def test_heatmap_and_peak_extraction() -> None:
    heatmap = build_heatmap([[32.0, 40.0, 16.0, 16.0]], image_size=64, stride=4)
    assert heatmap.shape == (16, 16)
    assert float(heatmap.max()) == 1.0
    peaks = extract_peaks(torch.from_numpy(heatmap), threshold=0.9)
    assert len(peaks) == 1
    assert peaks[0][:2] == (8.0, 10.0)


def test_model_shape_and_backward() -> None:
    model = PipeCenterNet()
    image = torch.randn(2, 3, 64, 64)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 16, 16] = 1.0
    logits = model(image)
    assert logits.shape == target.shape
    loss = modified_focal_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()


def test_geometry_flags_one_gap() -> None:
    centers = [
        (0, 0),
        (10, 0),
        (20, 0),
        (40, 0),
        (5, 9),
        (15, 9),
        (25, 9),
        (35, 9),
    ]
    result = analyze_lattice(centers)
    assert result["detected_count"] == 8
    assert result["possible_gap_count"] == 1
    possible_x, possible_y = result["possible_gap_points"][0]
    assert abs(possible_x - 30.0) < 0.1
    assert abs(possible_y) < 0.1


def _write_coco_split(
    split_dir: Path,
    image_specs: list[tuple[str, int]],
    duplicate_pixels: bool = False,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    annotation_id = 0
    for image_id, (name, box_count) in enumerate(image_specs):
        image = np.full((64, 64, 3), 30 if duplicate_pixels else image_id * 10, np.uint8)
        cv2.imwrite(str(split_dir / name), image)
        images.append({"id": image_id, "file_name": name, "width": 64, "height": 64})
        for box_index in range(box_count):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": box_index % 2,
                    "bbox": [4 + 12 * box_index, 10, 8, 8],
                    "area": 64,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 0, "name": "pipe"},
            {"id": 1, "name": "tube"},
        ],
    }
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_prepare_dataset_discovers_and_deduplicates(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _write_coco_split(
        root / "export_a" / "train",
        [("a.jpg", 1), ("b.jpg", 2)],
    )
    _write_coco_split(
        root / "export_b" / "valid",
        [("c.jpg", 3)],
    )
    report = prepare_dataset(
        root,
        tmp_path / "processed",
        train_ratio=0.6,
        val_ratio=0.2,
        preview_count=2,
    )
    assert report["coco_json_count"] == 2
    assert report["records_before_deduplication"] == 3
    # a.jpg and c.jpg contain identical pixels. The copy with three boxes wins.
    assert report["unique_images"] == 2
    assert report["duplicate_copies_removed"] == 1
    assert report["annotation_conflict_count"] == 1
    assert report["total_centers"] == 5
    assert (tmp_path / "processed" / "train.jsonl").is_file()
    assert (tmp_path / "processed" / "val.jsonl").is_file()
    assert (tmp_path / "processed" / "test.jsonl").is_file()
