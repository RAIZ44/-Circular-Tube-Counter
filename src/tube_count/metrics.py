from __future__ import annotations

import numpy as np

from tube_count.targets import box_iou_matrix


def count_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError("pred and truth must have the same shape")
    err = pred - truth
    mae = float(np.mean(np.abs(err))) if len(err) else 0.0
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else 0.0
    exact = float(np.mean(pred == truth) * 100.0) if len(err) else 0.0
    over = float(np.mean(err > 0) * 100.0) if len(err) else 0.0
    under = float(np.mean(err < 0) * 100.0) if len(err) else 0.0
    return {
        "n": int(len(err)),
        "mae": mae,
        "rmse": rmse,
        "exact_match_pct": exact,
        "overcount_pct": over,
        "undercount_pct": under,
        "mean_pred": float(pred.mean()) if len(err) else 0.0,
        "mean_gt": float(truth.mean()) if len(err) else 0.0,
    }


def detection_f1(
    pred_boxes_list: list[np.ndarray],
    gt_boxes_list: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    tp = fp = fn = 0
    for pred, gt in zip(pred_boxes_list, gt_boxes_list, strict=True):
        pred = np.asarray(pred, dtype=np.float32)
        gt = np.asarray(gt, dtype=np.float32)
        pred = pred.reshape(0, 4) if pred.size == 0 else pred.reshape(-1, 4)
        gt = gt.reshape(0, 4) if gt.size == 0 else gt.reshape(-1, 4)
        if len(pred) == 0 and len(gt) == 0:
            continue
        if len(pred) == 0:
            fn += len(gt)
            continue
        if len(gt) == 0:
            fp += len(pred)
            continue
        iou = box_iou_matrix(pred, gt)
        matched_gt = set()
        matched_pr = set()
        pairs = [
            (float(iou[i, j]), i, j)
            for i in range(iou.shape[0])
            for j in range(iou.shape[1])
            if iou[i, j] >= iou_threshold
        ]
        pairs.sort(reverse=True)
        for _, i, j in pairs:
            if i in matched_pr or j in matched_gt:
                continue
            matched_pr.add(i)
            matched_gt.add(j)
            tp += 1
        fp += len(pred) - len(matched_pr)
        fn += len(gt) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "iou_threshold": float(iou_threshold),
    }
