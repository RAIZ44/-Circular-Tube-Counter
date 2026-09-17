from __future__ import annotations

import argparse
import json

from pipe_counter.prepare import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Roboflow COCO exports for center-heatmap training."
    )
    parser.add_argument("--data-root", required=True, help="Parent of all COCO exports")
    parser.add_argument("--output-dir", default="data/processed_v2")
    parser.add_argument("--classes", nargs="+", default=["pipe", "tube"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--duplicate-policy",
        choices=["max_annotations", "first"],
        default="max_annotations",
    )
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument(
        "--no-perceptual-grouping",
        action="store_true",
        help="Disable perceptual links; source-name and exact-hash grouping remain enforced.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        classes=args.classes,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        duplicate_policy=args.duplicate_policy,
        preview_count=args.preview_count,
        group_near_duplicates=not args.no_perceptual_grouping,
    )
    summary = {
        key: report[key]
        for key in (
            "coco_json_count",
            "records_before_deduplication",
            "unique_images",
            "duplicate_copies_removed",
            "annotation_conflict_count",
            "possible_near_duplicate_group_count",
            "negative_image_count",
            "total_centers",
            "centers_per_image",
            "splits",
        )
    }
    print(json.dumps(summary, indent=2))
    print(f"\nFull report: {args.output_dir}/dataset_report.json")
    print(f"Review label previews: {args.output_dir}/previews")


if __name__ == "__main__":
    main()
