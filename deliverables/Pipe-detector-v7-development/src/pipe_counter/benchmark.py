"""Score reviewed field counts against fixed model predictions."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path

from .prepare import IMAGE_EXTENSIONS, perceptual_group_key, sha256_file
from .splitting import source_key
from .utils import read_jsonl, resolve_manifest_path


def create_review_sheet(image_root: Path, output: Path) -> int:
    images = sorted(p for p in image_root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
    if not images:
        raise ValueError("No supported images found")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects manually entered counts on subsequent runs.
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "sha256", "true_count", "reviewed", "scene_id", "notes"])
        writer.writeheader()
        for image in images:
            writer.writerow({"image": Path(os.path.relpath(image.resolve(), output.resolve().parent)).as_posix(),
                             "sha256": sha256_file(image), "true_count": "", "reviewed": "", "scene_id": "", "notes": ""})
    return len(images)


def _nonnegative_count(value: str) -> int:
    value = value.strip()
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"Counts must be nonnegative integers, got {value!r}")
    return int(value)


def score_reviewed_counts(review_csv: Path, predictions_csv: Path, development_manifests: list[Path]) -> dict:
    if not development_manifests:
        raise ValueError("Supply the training and validation manifests used by the checkpoint")
    with review_csv.open(newline="", encoding="utf-8-sig") as handle:
        reviewed = list(csv.DictReader(handle))
    with predictions_csv.open(newline="", encoding="utf-8-sig") as handle:
        predictions = list(csv.DictReader(handle))
    if not reviewed:
        raise ValueError("Review sheet is empty")
    known_hashes, known_sources, known_perceptual = set(), set(), set()
    for manifest in development_manifests:
        for record in read_jsonl(manifest):
            known_hashes.add(record["sha256"])
            known_sources.update(record.get("source_keys") or [source_key(record["source_file_name"])])
            if record.get("perceptual_group"):
                known_perceptual.add(record["perceptual_group"])
    predicted = {}
    for row in predictions:
        path = resolve_manifest_path(row["image"], predictions_csv.resolve())
        if path in predicted:
            raise ValueError(f"Duplicate prediction: {path}")
        predicted[path] = _nonnegative_count(row["detected_count"])
    details, seen_paths, seen_hashes = [], set(), set()
    for row in reviewed:
        path = resolve_manifest_path(row["image"], review_csv.resolve())
        if row.get("reviewed", "").strip().lower() not in {"yes", "true", "1"}:
            raise ValueError(f"Manual review not confirmed: {path.name}")
        if not row.get("scene_id", "").strip():
            raise ValueError(f"scene_id is required: {path.name}")
        truth = _nonnegative_count(row.get("true_count", ""))
        digest = sha256_file(path)
        if digest != row.get("sha256"):
            raise ValueError(f"Image changed after review sheet creation: {path.name}")
        if path in seen_paths or digest in seen_hashes:
            raise ValueError(f"Duplicate benchmark image: {path.name}")
        if (digest in known_hashes or source_key(path.name) in known_sources
                or perceptual_group_key(path) in known_perceptual):
            raise ValueError(f"Benchmark overlaps model development data: {path.name}")
        if path not in predicted:
            raise ValueError(f"Missing prediction: {path.name}")
        seen_paths.add(path)
        seen_hashes.add(digest)
        details.append({"image": row["image"], "scene_id": row["scene_id"], "true_count": truth,
                        "predicted_count": predicted[path], "error": predicted[path] - truth})
    if set(predicted) != seen_paths:
        raise ValueError("Predictions contain images outside the reviewed benchmark")
    errors = [row["error"] for row in details]
    n = len(errors)
    return {
        "images": n, "scenes": len({row["scene_id"] for row in details}),
        "exact_count_accuracy": sum(e == 0 for e in errors) / n,
        "within_one_accuracy": sum(abs(e) <= 1 for e in errors) / n,
        "mean_absolute_error": sum(abs(e) for e in errors) / n,
        "root_mean_squared_error": math.sqrt(sum(e * e for e in errors) / n),
        "mean_signed_error": sum(errors) / n,
        "development_manifests": [str(p.resolve()) for p in development_manifests],
        "limitations": "Automated overlap checks cannot prove scene independence; verify capture sessions manually. No threshold tuning is performed.",
        "per_image": details,
    }
