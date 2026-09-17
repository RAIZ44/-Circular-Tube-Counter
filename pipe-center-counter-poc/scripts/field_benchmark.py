from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipe_counter.benchmark import create_review_sheet, score_reviewed_counts
from pipe_counter.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a field-count review sheet or score reviewed predictions")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--images", type=Path, required=True)
    init.add_argument("--output", type=Path, default=Path("data/field_benchmark/review.csv"))
    score = commands.add_parser("score")
    score.add_argument("--review", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--development-manifests", type=Path, nargs="+", required=True,
                       help="Training AND validation manifests used by this checkpoint")
    score.add_argument("--output", type=Path, default=Path("runs/field_benchmark/metrics.json"))
    args = parser.parse_args()
    if args.command == "init":
        print(f"Created {create_review_sheet(args.images, args.output)} review rows in {args.output}")
    else:
        result = score_reviewed_counts(args.review, args.predictions, args.development_manifests)
        save_json(args.output, result)
        print(json.dumps({key: value for key, value in result.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
