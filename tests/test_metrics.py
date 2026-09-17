from __future__ import annotations

from pathlib import Path

import numpy as np

from tube_count.metrics import count_metrics, detection_f1


def test_count_metrics_perfect() -> None:
    m = count_metrics(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["exact_match_pct"] == 100.0
    assert m["overcount_pct"] == 0.0
    assert m["undercount_pct"] == 0.0


def test_count_metrics_over_under() -> None:
    m = count_metrics(np.array([3, 0, 2]), np.array([1, 2, 2]))
    assert m["mae"] == (2 + 2 + 0) / 3
    assert m["overcount_pct"] == pytest_pct(1, 3)
    assert m["undercount_pct"] == pytest_pct(1, 3)


def pytest_pct(n: int, d: int) -> float:
    return n / d * 100.0


def test_detection_f1_identity() -> None:
    boxes = [np.array([[10, 10, 30, 30], [50, 50, 80, 80]], dtype=np.float32)]
    m = detection_f1(boxes, boxes, iou_threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_detection_f1_empty_pred() -> None:
    pred = [np.zeros((0, 4), dtype=np.float32)]
    gt = [np.array([[0, 0, 10, 10]], dtype=np.float32)]
    m = detection_f1(pred, gt, iou_threshold=0.5)
    assert m["tp"] == 0
    assert m["fn"] == 1
    assert m["recall"] == 0.0
