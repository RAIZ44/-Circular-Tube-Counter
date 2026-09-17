from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from pipe_counter.data import letterbox, normalize_image
from pipe_counter.geometry import analyze_lattice
from pipe_counter.model import load_checkpoint_model
from pipe_counter.peaks import extract_peaks
from pipe_counter.utils import autocast_enabled, resolve_device, save_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count pipe ends in images")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image or directory of images")
    parser.add_argument("--output-dir", default="runs/predictions")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--peak-kernel", type=int, default=3)
    parser.add_argument("--disable-geometry", action="store_true")
    parser.add_argument("--save-heatmaps", action="store_true")
    return parser.parse_args()


def discover_images(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"No image or image directory found: {input_path}")


def map_peaks_to_original(
    peaks: list[tuple[float, float, float]],
    stride: int,
    meta: dict[str, float],
) -> list[tuple[float, float, float]]:
    mapped: list[tuple[float, float, float]] = []
    for x, y, confidence in peaks:
        original_x = (x * stride - meta["pad_x"]) / meta["scale"]
        original_y = (y * stride - meta["pad_y"]) / meta["scale"]
        if (
            0 <= original_x < meta["original_width"]
            and 0 <= original_y < meta["original_height"]
        ):
            mapped.append((original_x, original_y, confidence))
    return mapped


def draw_prediction(
    image: np.ndarray,
    centers: list[tuple[float, float, float]],
    geometry: dict,
    threshold: float,
) -> np.ndarray:
    output = image.copy()
    for center_x, center_y, confidence in centers:
        point = (int(round(center_x)), int(round(center_y)))
        cv2.circle(output, point, 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(output, point, 8, (0, 70, 0), 1, cv2.LINE_AA)

    for center_x, center_y in geometry.get("possible_gap_points", []):
        x, y = int(round(center_x)), int(round(center_y))
        cv2.line(output, (x - 8, y - 8), (x + 8, y + 8), (0, 215, 255), 3)
        cv2.line(output, (x - 8, y + 8), (x + 8, y - 8), (0, 215, 255), 3)

    lines = [
        f"Detected: {len(centers)}",
        f"Possible gaps: {geometry.get('possible_gap_count', 0)}",
        f"Threshold: {threshold:.3f}",
    ]
    font_scale = max(0.65, min(1.1, output.shape[1] / 1200.0))
    line_height = int(36 * font_scale)
    panel_width = int(340 * font_scale)
    panel_height = line_height * len(lines) + 24
    overlay = output.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, output, 0.35, 0, output)
    for index, text in enumerate(lines):
        cv2.putText(
            output,
            text,
            (20, 8 + line_height * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", 512))
    stride = int(config.get("stride", 4))
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint.get("threshold", 0.3))
    )

    input_path = Path(args.input).expanduser().resolve()
    images = discover_images(input_path)
    if not images:
        raise RuntimeError(f"No supported images found below {input_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = output_dir / "heatmaps"
    if args.save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with torch.inference_mode():
        for index, image_path in enumerate(images):
            original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if original is None:
                print(f"Skipping unreadable image: {image_path}")
                continue
            prepared, _, meta = letterbox(original, [], image_size)
            tensor = normalize_image(prepared)[None, ...].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast_enabled(device),
            ):
                heatmap = model(tensor).sigmoid()[0, 0].cpu()
            peaks = extract_peaks(
                heatmap,
                threshold=threshold,
                kernel_size=args.peak_kernel,
            )
            centers = map_peaks_to_original(peaks, stride, meta)
            points = [(center[0], center[1]) for center in centers]
            geometry = (
                analyze_lattice(points)
                if not args.disable_geometry
                else {
                    "detected_count": len(points),
                    "possible_gap_count": 0,
                    "possible_gap_points": [],
                    "rows": [],
                }
            )
            rendered = draw_prediction(original, centers, geometry, threshold)
            output_name = f"{index:04d}_{image_path.stem}_count_{len(centers)}.jpg"
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), rendered)

            if args.save_heatmaps:
                normalized = np.clip(heatmap.numpy() * 255.0, 0, 255).astype(np.uint8)
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                cv2.imwrite(str(heatmap_dir / output_name), colored)

            detail_path = output_dir / f"{index:04d}_{image_path.stem}.json"
            save_json(
                detail_path,
                {
                    "image": str(image_path),
                    "detected_count": len(centers),
                    "threshold": threshold,
                    "centers": [
                        {"x": x, "y": y, "confidence": confidence}
                        for x, y, confidence in centers
                    ],
                    "geometry": geometry,
                    "rendered_image": str(output_path),
                },
            )
            rows.append(
                {
                    "image": str(image_path),
                    "detected_count": len(centers),
                    "possible_gap_count": geometry.get("possible_gap_count", 0),
                    "output": str(output_path),
                }
            )
            print(
                f"{image_path.name}: detected={len(centers)} "
                f"possible_gaps={geometry.get('possible_gap_count', 0)}"
            )

    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "detected_count", "possible_gap_count", "output"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Predictions: {output_dir}")


if __name__ == "__main__":
    main()
