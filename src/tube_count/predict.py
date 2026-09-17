from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tube_count.evaluate import load_checkpoint
from tube_count.targets import decode_heatmap
from tube_count.utils import dump_json, resolve_device
from tube_count.viz import overlay_boxes

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _load_rgb(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        if rgb.size != (image_size, image_size):
            rgb = rgb.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(rgb)


@torch.no_grad()
def predict_image(
    model,
    image: np.ndarray,
    device: torch.device,
    image_size: int,
    stride: int,
    conf_threshold: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div(255.0)
    tensor = tensor.unsqueeze(0).to(device)
    hm, rad = model(tensor)
    boxes, scores = decode_heatmap(hm[0], rad[0], image_size, stride, conf_threshold)
    return int(len(boxes)), boxes, scores


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(source)


def run_predict(
    input_path: str | Path,
    checkpoint: str | Path,
    out_dir: str | Path | None = None,
    conf_threshold: float | None = None,
    device_name: str = "auto",
    image_size: int | None = None,
) -> list[dict]:
    device = resolve_device(device_name)
    model, ckpt = load_checkpoint(checkpoint, device)
    cfg = ckpt.get("config") or {}
    size = int(image_size or cfg.get("generate", {}).get("image_size", 256))
    stride = int(cfg.get("model", {}).get("stride", 4))
    thr = float(conf_threshold if conf_threshold is not None else ckpt.get("conf_threshold", 0.3))
    paths = collect_images(Path(input_path))
    dest = Path(out_dir) if out_dir else None
    if dest:
        dest.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for path in paths:
        image = _load_rgb(path, size)
        count, boxes, scores = predict_image(model, image, device, size, stride, thr)
        rec = {"file": str(path), "count": count, "conf_threshold": thr}
        rows.append(rec)
        print(f"{path}: {count}")
        if dest is not None:
            annotated = dest / f"{path.stem}_pred{count}.png"
            overlay_boxes(image, np.zeros((0, 4)), boxes, gt_count=None, pred_count=count, path=annotated)
            rec["annotated"] = str(annotated)
    if dest is not None:
        dump_json(dest / "predictions.json", rows)
        with (dest / "predictions.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["file", "count", "conf_threshold", "annotated"])
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Count circular tubes in an image or folder")
    p.add_argument("--input", required=True, help="Image file or directory")
    p.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt")
    p.add_argument("--out", default=None, help="Optional directory for annotated images + CSV")
    p.add_argument("--conf-threshold", type=float, default=None)
    p.add_argument("--device", default="auto")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_predict(args.input, args.checkpoint, args.out, args.conf_threshold, args.device)


if __name__ == "__main__":
    main()
