from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .utils import save_json, write_jsonl
from .splitting import assign_groups, source_key, split_overlap_report


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_group_key(path: Path) -> str:
    """Stable dHash grouping key used to keep near duplicates in one split.

    This does not delete images. It only reduces train/test leakage from resized
    or re-encoded copies that have identical low-resolution gradients.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Unable to decode image: {path}")
    height, width = image.shape[:2]
    dhash_image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    differences = dhash_image[:, 1:] > dhash_image[:, :-1]
    dhash_value = 0
    for bit in differences.flatten():
        dhash_value = (dhash_value << 1) | int(bit)

    phash_image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(phash_image.astype(np.float32))[:8, :8]
    median = float(np.median(dct.flatten()[1:]))
    phash_value = 0
    for bit in (dct > median).flatten():
        phash_value = (phash_value << 1) | int(bit)

    aspect_bucket = round(width / max(height, 1), 2)
    mean_bucket = int(round(float(image.mean()) / 2.0))
    return (
        f"{aspect_bucket:.2f}:{mean_bucket:03d}:"
        f"{dhash_value:016x}:{phash_value:016x}"
    )


def discover_coco_jsons(data_root: Path) -> list[Path]:
    found: list[Path] = []
    for path in sorted(data_root.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if all(key in payload for key in ("images", "annotations", "categories")):
                found.append(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return found


def _resolve_image(json_path: Path, file_name: str) -> Path | None:
    normalized = file_name.replace("\\", "/")
    candidates = [
        json_path.parent / normalized,
        json_path.parent / Path(normalized).name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    basename = Path(normalized).name
    matches = list(json_path.parent.rglob(basename))
    return matches[0].resolve() if len(matches) == 1 else None


def _clip_box(
    bbox: Sequence[float], width: int, height: int
) -> list[float] | None:
    if len(bbox) != 4:
        return None
    x, y, box_w, box_h = (float(value) for value in bbox)
    values = np.asarray([x, y, box_w, box_h], dtype=np.float64)
    if not np.isfinite(values).all() or box_w <= 0 or box_h <= 0:
        return None

    x1 = float(np.clip(x, 0, width))
    y1 = float(np.clip(y, 0, height))
    x2 = float(np.clip(x + box_w, 0, width))
    y2 = float(np.clip(y + box_h, 0, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]


def load_coco_records(
    json_path: Path, target_classes: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    categories = {
        int(category["id"]): str(category["name"]).strip().lower()
        for category in payload["categories"]
    }
    selected_ids = {
        category_id
        for category_id, name in categories.items()
        if name in target_classes
    }

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ignored_class_annotations = 0
    invalid_annotations = 0
    for annotation in payload["annotations"]:
        try:
            category_id = int(annotation["category_id"])
            image_id = int(annotation["image_id"])
        except (KeyError, TypeError, ValueError):
            invalid_annotations += 1
            continue
        if category_id not in selected_ids:
            ignored_class_annotations += 1
            continue
        annotations_by_image[image_id].append(annotation)

    records: list[dict[str, Any]] = []
    missing_images: list[str] = []
    for image in payload["images"]:
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        image_path = _resolve_image(json_path, file_name)
        if image_path is None:
            missing_images.append(file_name)
            continue

        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        if width <= 0 or height <= 0:
            loaded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if loaded is None:
                missing_images.append(file_name)
                continue
            height, width = loaded.shape[:2]

        boxes: list[list[float]] = []
        for annotation in annotations_by_image.get(image_id, []):
            clipped = _clip_box(annotation.get("bbox", []), width, height)
            if clipped is None:
                invalid_annotations += 1
            else:
                boxes.append(clipped)

        records.append(
            {
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "boxes": boxes,
                "source_json": str(json_path.resolve()),
                "source_image_id": image_id,
                "source_file_name": file_name,
            }
        )

    stats = {
        "json": str(json_path.resolve()),
        "categories": categories,
        "selected_category_ids": sorted(selected_ids),
        "records": len(records),
        "annotations": sum(len(record["boxes"]) for record in records),
        "ignored_class_annotations": ignored_class_annotations,
        "invalid_annotations": invalid_annotations,
        "missing_images": missing_images[:50],
        "missing_image_count": len(missing_images),
    }
    return records, stats


def _assign_splits(
    records: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    group_near_duplicates: bool,
) -> dict[str, list[dict[str, Any]]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must satisfy train > 0, val >= 0, and train + val < 1")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    assign_groups(records, use_perceptual=group_near_duplicates)
    for record in records:
        group_key = record["split_group"]
        groups[group_key].append(record)
    grouped_records = list(groups.values())
    random.Random(seed).shuffle(grouped_records)

    total = len(records)
    targets = {
        "train": int(round(total * train_ratio)),
        "val": int(round(total * val_ratio)),
    }
    targets["test"] = total - targets["train"] - targets["val"]
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for group in grouped_records:
        deficits = {
            name: targets[name] - len(splits[name])
            for name in ("train", "val", "test")
        }
        destination = max(deficits, key=lambda name: (deficits[name], targets[name]))
        splits[destination].extend(group)
    return splits


def _draw_previews(
    records: list[dict[str, Any]], output_dir: Path, seed: int, limit: int = 12
) -> list[str]:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(records)
    random.Random(seed).shuffle(candidates)
    outputs: list[str] = []

    for index, record in enumerate(candidates[:limit]):
        image = cv2.imread(record["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        for cx, cy, box_w, box_h in record["boxes"]:
            radius = max(3, int(round(min(box_w, box_h) * 0.08)))
            cv2.circle(image, (round(cx), round(cy)), radius, (0, 255, 0), -1)
        cv2.putText(
            image,
            f"centers: {len(record['boxes'])}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"centers: {len(record['boxes'])}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        output = preview_dir / f"preview_{index:02d}.jpg"
        cv2.imwrite(str(output), image)
        outputs.append(str(output.resolve()))
    return outputs


def prepare_dataset(
    data_root: str | Path,
    output_dir: str | Path,
    classes: Sequence[str] = ("pipe", "tube"),
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    duplicate_policy: str = "max_annotations",
    preview_count: int = 12,
    group_near_duplicates: bool = True,
) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    target_classes = {name.strip().lower() for name in classes}
    coco_jsons = discover_coco_jsons(root)
    if not coco_jsons:
        raise RuntimeError(f"No COCO annotation JSON files found below {root}")

    all_records: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    for json_path in coco_jsons:
        records, stats = load_coco_records(json_path, target_classes)
        all_records.extend(records)
        source_reports.append(stats)

    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unreadable_images: list[str] = []
    for record in all_records:
        try:
            image_path = Path(record["image_path"])
            image_hash = sha256_file(image_path)
            record["perceptual_group"] = perceptual_group_key(image_path)
        except OSError:
            unreadable_images.append(record["image_path"])
            continue
        record["sha256"] = image_hash
        by_hash[image_hash].append(record)

    chosen: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    duplicate_copies = 0
    for image_hash, candidates in by_hash.items():
        duplicate_copies += max(0, len(candidates) - 1)
        if duplicate_policy == "max_annotations":
            selected = max(
                candidates,
                key=lambda record: (len(record["boxes"]), record["source_json"]),
            )
        elif duplicate_policy == "first":
            selected = candidates[0]
        else:
            raise ValueError(f"Unsupported duplicate policy: {duplicate_policy}")

        counts = sorted({len(candidate["boxes"]) for candidate in candidates})
        signatures = {
            tuple(sorted(tuple(round(value, 3) for value in box) for box in candidate["boxes"]))
            for candidate in candidates
        }
        if len(signatures) > 1:
            conflicts.append(
                {
                    "sha256": image_hash,
                    "annotation_counts": counts,
                    "selected_count": len(selected["boxes"]),
                    "selected_source": selected["source_json"],
                    "copies": [candidate["image_path"] for candidate in candidates],
                }
            )
        selected["source_keys"] = sorted({source_key(candidate["source_file_name"]) for candidate in candidates})
        selected["annotation_conflict"] = len(signatures) > 1
        chosen.append(selected)

    by_perceptual_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in chosen:
        by_perceptual_group[record["perceptual_group"]].append(record)
    possible_near_duplicates = [
        {
            "perceptual_group": group_key,
            "image_count": len(group),
            "images": [record["image_path"] for record in group],
        }
        for group_key, group in by_perceptual_group.items()
        if len(group) > 1
    ]

    counts = [len(record["boxes"]) for record in chosen]
    if not counts or sum(counts) == 0:
        available = sorted(
            {
                name
                for source in source_reports
                for name in source["categories"].values()
            }
        )
        raise RuntimeError(
            "No target annotations were found. "
            f"Requested classes={sorted(target_classes)}, available classes={available}"
        )

    splits = _assign_splits(
        chosen, seed, train_ratio, val_ratio, group_near_duplicates
    )
    overlap = split_overlap_report(splits)
    enforced = ("sha256", "source_keys", "split_group") + (("perceptual_group",) if group_near_duplicates else ())
    if any(count for field in enforced for count in overlap[field].values()):
        raise RuntimeError("Related images crossed dataset splits")
    if len(chosen) >= 3 and any(not records for records in splits.values()):
        raise RuntimeError(
            "At least one split is empty because related-image groups are unusually large. "
            "Review source identities and perceptual matches before changing grouping."
        )
    for split_name, records in splits.items():
        for record in records:
            record["split"] = split_name
        write_jsonl(output / f"{split_name}.jsonl", records)

    previews = _draw_previews(chosen, output, seed, preview_count)
    save_json(output / "annotation_conflicts.json", conflicts)
    report = {
        "manifest_schema": 2,
        "path_base": "manifest_directory",
        "grouping": "transitive source name + exact hash + optional perceptual hash",
        "split_overlap": overlap,
        "independent_field_benchmark": False,
        "data_root": str(root),
        "target_classes": sorted(target_classes),
        "coco_json_count": len(coco_jsons),
        "source_reports": source_reports,
        "records_before_deduplication": len(all_records),
        "unique_images": len(chosen),
        "duplicate_copies_removed": duplicate_copies,
        "annotation_conflict_count": len(conflicts),
        "annotation_conflicts": conflicts[:100],
        "possible_near_duplicate_group_count": len(possible_near_duplicates),
        "possible_near_duplicate_groups": possible_near_duplicates[:100],
        "unreadable_images": unreadable_images[:50],
        "unreadable_image_count": len(unreadable_images),
        "negative_image_count": sum(count == 0 for count in counts),
        "total_centers": int(sum(counts)),
        "centers_per_image": {
            "min": int(min(counts, default=0)),
            "mean": float(np.mean(counts)) if counts else 0.0,
            "median": float(np.median(counts)) if counts else 0.0,
            "max": int(max(counts, default=0)),
        },
        "splits": {name: len(records) for name, records in splits.items()},
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "duplicate_policy": duplicate_policy,
        "group_near_duplicates": group_near_duplicates,
        "previews": previews,
    }
    save_json(output / "dataset_report.json", report)
    return report
