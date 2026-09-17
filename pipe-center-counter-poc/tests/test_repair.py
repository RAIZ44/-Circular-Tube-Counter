from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipe_counter.benchmark import create_review_sheet, score_reviewed_counts
from pipe_counter.prepare import _assign_splits, sha256_file
from pipe_counter.splitting import assign_groups, source_key, split_overlap_report
from pipe_counter.utils import read_jsonl, write_jsonl


def test_transitive_groups_keep_duplicate_aliases_and_variants_together():
    records = [
        {"sha256": "a", "source_file_name": "first.rf.1.jpg", "source_keys": ["first", "alias"], "perceptual_group": "p1"},
        {"sha256": "b", "source_file_name": "alias.rf.2.jpg", "perceptual_group": "p2"},
        {"sha256": "c", "source_file_name": "renamed.rf.3.jpg", "perceptual_group": "p2"},
    ]
    assign_groups(records)
    assert len({r["split_group"] for r in records}) == 1
    independent = [{"sha256": str(i), "source_file_name": f"other{i}.jpg", "perceptual_group": f"x{i}"} for i in range(20)]
    splits = _assign_splits(records + independent, 42, .6, .2, True)
    assert all(splits.values())
    assert not any(n for pairs in split_overlap_report(splits).values() for n in pairs.values())
    assert source_key("PHOTO_jpg.rf.123.jpg") == source_key("photo_jpg.rf.456.jpg")


def test_manifests_survive_moving_whole_project(tmp_path):
    original = tmp_path / "original"
    image = original / "pipe-datasets" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    manifest = original / "project" / "data" / "train.jsonl"
    write_jsonl(manifest, [{"image_path": str(image)}])
    assert not Path(json.loads(manifest.read_text())["image_path"]).is_absolute()
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    loaded = read_jsonl(moved / "project" / "data" / "train.jsonl")
    assert Path(loaded[0]["image_path"]) == moved / "pipe-datasets" / "a.jpg"


def test_legacy_account_path_resolves_only_in_sibling_dataset(tmp_path):
    image = tmp_path / "pipe-datasets" / "train" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    manifest = tmp_path / "project" / "data" / "train.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"image_path": r"C:\Users\old\projects\pipe-datasets\train\a.jpg"}))
    assert Path(read_jsonl(manifest)[0]["image_path"]) == image


def test_field_benchmark_requires_review_and_rejects_training_overlap(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    image = images / "field.jpg"
    cv2.imwrite(str(image), np.full((16, 16, 3), 73, np.uint8))
    review = tmp_path / "review.csv"
    assert create_review_sheet(images, review) == 1
    with pytest.raises(FileExistsError):
        create_review_sheet(images, review)
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("image,detected_count\nimages/field.jpg,6\n")
    development = tmp_path / "train.jsonl"
    write_jsonl(development, [{"sha256": "different", "source_file_name": "other.jpg"}])
    with pytest.raises(ValueError, match="Manual review"):
        score_reviewed_counts(review, predictions, [development])
    with review.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0].update(true_count="5", reviewed="yes", scene_id="session-1")
    with review.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = score_reviewed_counts(review, predictions, [development])
    assert result["within_one_accuracy"] == 1
    assert result["exact_count_accuracy"] == 0
    assert result["mean_absolute_error"] == 1
    write_jsonl(development, [{"sha256": sha256_file(image), "source_file_name": "renamed.jpg"}])
    with pytest.raises(ValueError, match="overlaps"):
        score_reviewed_counts(review, predictions, [development])
