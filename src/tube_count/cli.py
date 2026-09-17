from __future__ import annotations

import argparse

from tube_count.baselines import main as baselines_main
from tube_count.evaluate import main as evaluate_main
from tube_count.generate import main as generate_main
from tube_count.predict import main as predict_main
from tube_count.train import main as train_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tube-count", description="Circular tube counting")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate", help="Write synthetic YOLO dataset")
    sub.add_parser("train", help="Train TubeNet")
    sub.add_parser("evaluate", help="Evaluate on the test split")
    sub.add_parser("predict", help="Run inference on an image or folder")
    sub.add_parser("baseline", help="Run mean-count and Hough baselines")
    args, rest = parser.parse_known_args(argv)
    dispatch = {
        "generate": generate_main,
        "train": train_main,
        "evaluate": evaluate_main,
        "predict": predict_main,
        "baseline": baselines_main,
    }
    dispatch[args.cmd](rest)


if __name__ == "__main__":
    main()
