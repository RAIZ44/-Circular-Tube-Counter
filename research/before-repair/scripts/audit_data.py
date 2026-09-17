from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipe_counter.utils import read_jsonl, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit center density, heatmap collisions, and label previews"
    )
    parser.add_argument("--manifest-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/audit")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--preview-per-band", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def percentile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    return {
        f"p{value:02d}": float(np.percentile(array, value)) for value in percentiles
    }


def transformed_boxes(record: dict[str, Any], image_size: int) -> np.ndarray:
    scale = min(image_size / record["width"], image_size / record["height"])
    resized_width = int(round(record["width"] * scale))
    resized_height = int(round(record["height"] * scale))
    pad_x = (image_size - resized_width) // 2
    pad_y = (image_size - resized_height) // 2
    boxes = np.asarray(record["boxes"], dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    transformed = boxes.copy()
    transformed[:, 0] = transformed[:, 0] * scale + pad_x
    transformed[:, 1] = transformed[:, 1] * scale + pad_y
    transformed[:, 2:] *= scale
    return transformed


def collision_stats(records: list[dict[str, Any]], image_size: int, stride: int) -> dict:
    collision_images = 0
    excess_centers = 0
    total_centers = 0
    for record in records:
        boxes = transformed_boxes(record, image_size)
        total_centers += len(boxes)
        if len(boxes) < 2:
            continue
        grid = np.rint(boxes[:, :2] / stride).astype(np.int64)
        unique = len(np.unique(grid, axis=0))
        excess = len(grid) - unique
        if excess:
            collision_images += 1
            excess_centers += excess
    return {
        "stride": stride,
        "images_with_same_pixel_centers": collision_images,
        "excess_centers_on_occupied_pixels": excess_centers,
        "center_collision_rate": excess_centers / max(total_centers, 1),
    }


def nearest_neighbor_distances(points: np.ndarray) -> list[float]:
    if len(points) < 2:
        return []
    minimum = np.full(len(points), np.inf, dtype=np.float64)
    chunk_size = 256
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        deltas = points[start:end, None, :] - points[None, :, :]
        squared = np.sum(deltas * deltas, axis=2)
        rows = np.arange(end - start)
        squared[rows, np.arange(start, end)] = np.inf
        minimum[start:end] = np.sqrt(np.min(squared, axis=1))
    return minimum[np.isfinite(minimum)].tolist()


def choose_preview_records(
    records: list[dict[str, Any]], per_band: int, seed: int
) -> list[tuple[str, dict[str, Any]]]:
    positives = [record for record in records if record["boxes"]]
    ordered = sorted(positives, key=lambda record: len(record["boxes"]))
    if not ordered:
        return []
    rng = random.Random(seed)
    total = len(ordered)
    bands = {
        "sparse": ordered[: max(per_band * 3, total // 10)],
        "typical": ordered[max(0, total // 2 - total // 20) : max(1, total // 2 + total // 20)],
        "dense": ordered[max(0, int(total * 0.88)) : max(1, int(total * 0.95))],
        "extreme": ordered[-max(per_band * 3, min(30, total)) :],
    }
    selected: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()
    for band_name, candidates in bands.items():
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        if band_name == "extreme":
            shuffled = sorted(shuffled, key=lambda record: len(record["boxes"]), reverse=True)
        for record in shuffled:
            if record["sha256"] in used:
                continue
            selected.append((band_name, record))
            used.add(record["sha256"])
            if sum(name == band_name for name, _ in selected) >= per_band:
                break
    return selected


def render_audit_previews(
    selected: list[tuple[str, dict[str, Any]]], output_dir: Path
) -> list[dict[str, Any]]:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, (band, record) in enumerate(selected):
        image = cv2.imread(record["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        radius = max(2, int(round(min(image.shape[:2]) / 300)))
        for center_x, center_y, _, _ in record["boxes"]:
            cv2.circle(
                image,
                (int(round(center_x)), int(round(center_y))),
                radius,
                (0, 255, 0),
                -1,
                cv2.LINE_AA,
            )
        label = f"{band} | annotations: {len(record['boxes'])}"
        cv2.putText(image, label, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(image, label, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        output_path = preview_dir / f"{index:02d}_{band}_{len(record['boxes'])}.jpg"
        cv2.imwrite(str(output_path), image)
        rows.append(
            {
                "preview": str(output_path.resolve()),
                "band": band,
                "annotation_count": len(record["boxes"]),
                "source_image": record["image_path"],
                "manual_visible_count": "",
                "missing_labels": "",
                "false_labels": "",
                "notes": "",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for split in ("train", "val", "test"):
        path = manifest_dir / f"{split}.jsonl"
        records.extend(read_jsonl(path))
    if not records:
        raise RuntimeError("No prepared records found")

    counts: list[float] = []
    original_diameters: list[float] = []
    scaled_diameters: list[float] = []
    for record in records:
        counts.append(float(len(record["boxes"])))
        boxes = np.asarray(record["boxes"], dtype=np.float64)
        if boxes.size:
            original_diameters.extend(np.min(boxes[:, 2:4], axis=1).tolist())
        transformed = transformed_boxes(record, args.image_size)
        if transformed.size:
            scaled_diameters.extend(np.min(transformed[:, 2:4], axis=1).tolist())

    rng = random.Random(args.seed)
    positive_records = [record for record in records if len(record["boxes"]) >= 2]
    sampled = rng.sample(positive_records, min(300, len(positive_records)))
    sampled.extend(sorted(positive_records, key=lambda record: len(record["boxes"]), reverse=True)[:10])
    neighbor_distances: list[float] = []
    used: set[str] = set()
    for record in sampled:
        if record["sha256"] in used:
            continue
        used.add(record["sha256"])
        boxes = transformed_boxes(record, args.image_size)
        neighbor_distances.extend(nearest_neighbor_distances(boxes[:, :2]))

    selected = choose_preview_records(records, args.preview_per_band, args.seed)
    audit_rows = render_audit_previews(selected, output_dir)
    with (output_dir / "manual_label_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    report = {
        "images": len(records),
        "centers": int(sum(counts)),
        "target_image_size": args.image_size,
        "counts_per_image": percentile_summary(counts),
        "original_box_min_dimension_px": percentile_summary(original_diameters),
        "scaled_box_min_dimension_px": percentile_summary(scaled_diameters),
        "sampled_nearest_center_spacing_px": percentile_summary(neighbor_distances),
        "heatmap_collision_comparison": [
            collision_stats(records, args.image_size, stride=2),
            collision_stats(records, args.image_size, stride=4),
        ],
        "audit_preview_count": len(audit_rows),
        "manual_audit_csv": str((output_dir / "manual_label_audit.csv").resolve()),
    }
    save_json(output_dir / "density_report.json", report)
    print(json.dumps(report, indent=2))
    print(f"\nReview previews: {output_dir / 'previews'}")
    print(f"Record manual misses: {output_dir / 'manual_label_audit.csv'}")


if __name__ == "__main__":
    main()

