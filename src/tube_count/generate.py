"""Synthetic circular-tube image generator.

Writes a YOLO-format dataset under ``data/`` (see ``tube_count.yolo`` and
``data/README.md``). Real photos can replace these files without changing
train or eval as long as the same folder layout and label schema are kept.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from tube_count.config import Config, GenerateConfig
from tube_count.utils import dump_json, ensure_dir, set_seed
from tube_count.yolo import CLASS_NAME, write_yolo_label

SPLITS = ("train", "val", "test")
METAL_RGB = (
    (196, 198, 204),
    (168, 172, 180),
    (140, 148, 158),
    (186, 150, 110),
    (120, 128, 136),
    (210, 205, 198),
)


def _rgb(color: Sequence[int]) -> tuple[int, int, int]:
    return int(color[0]), int(color[1]), int(color[2])


def _render_background(size: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.integers(48, 170, size=3, dtype=np.int32)
    img = np.full((size, size, 3), base, dtype=np.uint8)
    blotch = rng.normal(0, 18, (size, size, 3))
    img = np.clip(img.astype(np.float32) + blotch, 0, 255).astype(np.uint8)
    n_streaks = int(rng.integers(4, 12))
    for _ in range(n_streaks):
        p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        p2 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        color = _rgb(rng.integers(20, 230, size=3))
        cv2.line(img, p1, p2, color, int(rng.integers(1, 6)), cv2.LINE_AA)
    n_blobs = int(rng.integers(3, 9))
    for _ in range(n_blobs):
        center = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        radius = int(rng.integers(12, 55))
        color = _rgb(rng.integers(30, 200, size=3))
        overlay = img.copy()
        cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, float(rng.uniform(0.15, 0.4)), img, 0.75, 0)
    return img


def _add_clutter(img: np.ndarray, rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    n = int(rng.integers(3, 10))
    for _ in range(n):
        color = _rgb(rng.integers(10, 245, size=3))
        kind = str(rng.choice(["rect", "poly", "line", "ellipse"]))
        if kind == "rect":
            x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
            x2, y2 = int(rng.integers(0, w)), int(rng.integers(0, h))
            thickness = int(rng.choice([-1, 1, 2, 3]))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        elif kind == "line":
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(img, p1, p2, color, int(rng.integers(1, 5)), cv2.LINE_AA)
        elif kind == "ellipse":
            center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            axes = (int(rng.integers(8, 40)), int(rng.integers(4, 18)))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(img, center, axes, angle, 0, 360, color, int(rng.choice([-1, 2])))
        else:
            pts = rng.integers(0, min(h, w), size=(rng.integers(3, 6), 2)).astype(np.int32)
            cv2.fillConvexPoly(img, pts, color)


def _draw_tube(img: np.ndarray, cx: int, cy: int, r_out: int, rng: np.random.Generator) -> None:
    wall = float(rng.uniform(0.18, 0.42))
    r_in = max(2, int(round(r_out * (1.0 - wall))))
    metal = _rgb(METAL_RGB[int(rng.integers(0, len(METAL_RGB)))])
    shade = float(rng.uniform(0.75, 1.08))
    metal = _rgb([int(np.clip(c * shade, 0, 255)) for c in metal])
    hole = _rgb([max(8, int(c * 0.22 + rng.integers(0, 18))) for c in metal])
    rim = _rgb([min(255, int(c + 45)) for c in metal])
    cv2.circle(img, (cx, cy), r_out, metal, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r_in, hole, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r_out, rim, max(1, r_out // 12), cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r_in, (18, 18, 20), 1, cv2.LINE_AA)
    spec_r = max(1, r_out // 7)
    spec = (int(cx - r_out * 0.28), int(cy - r_out * 0.28))
    cv2.circle(img, spec, spec_r, rim, -1, cv2.LINE_AA)


def _sample_tubes(
    size: int,
    count: int,
    cfg: GenerateConfig,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    tubes: list[tuple[int, int, int]] = []
    for _ in range(count):
        r = int(rng.integers(cfg.min_radius, cfg.max_radius + 1))
        placed = False
        for _try in range(60):
            margin = max(2, r // 4)
            cx = int(rng.integers(-margin, size + margin))
            cy = int(rng.integers(-margin, size + margin))
            if not tubes or float(rng.random()) < cfg.overlap_prob:
                tubes.append((cx, cy, r))
                placed = True
                break
            too_close = False
            for ox, oy, orad in tubes:
                dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                if dist < 0.75 * (r + orad):
                    too_close = True
                    break
            if not too_close:
                tubes.append((cx, cy, r))
                placed = True
                break
        if not placed:
            cx = int(rng.integers(0, size))
            cy = int(rng.integers(0, size))
            tubes.append((cx, cy, r))
    return tubes


def _apply_lighting(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    gx = float(rng.uniform(0.15, 0.85) * w)
    gy = float(rng.uniform(0.15, 0.85) * h)
    dist = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
    denom = float(dist.max()) or 1.0
    vignette = 0.62 + 0.5 * (1.0 - dist / denom)
    lit = img.astype(np.float32) * vignette[..., None]
    return np.clip(lit, 0, 255).astype(np.uint8)


def render_scene(cfg: GenerateConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    size = cfg.image_size
    img = _render_background(size, rng)
    if float(rng.random()) < cfg.clutter_prob:
        _add_clutter(img, rng)
    count = int(rng.integers(cfg.min_count, cfg.max_count + 1))
    tubes = _sample_tubes(size, count, cfg, rng)
    # Draw far-to-near so later tubes occlude earlier ones; all in-frame tubes are labeled.
    for cx, cy, r in tubes:
        _draw_tube(img, cx, cy, r, rng)
    img = _apply_lighting(img, rng)
    if cfg.noise_std > 0:
        noise = rng.normal(0, cfg.noise_std, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    boxes: list[list[float]] = []
    for cx, cy, r in tubes:
        x1, y1 = cx - r, cy - r
        x2, y2 = cx + r, cy + r
        if x2 <= 0 or y2 <= 0 or x1 >= size or y1 >= size:
            continue
        x1 = float(np.clip(x1, 0, size - 1))
        y1 = float(np.clip(y1, 0, size - 1))
        x2 = float(np.clip(x2, 0, size - 1))
        y2 = float(np.clip(y2, 0, size - 1))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        boxes.append([x1, y1, x2, y2])
    box_arr = np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
    return img, box_arr


def write_dataset_index(root: Path, image_size: int) -> None:
    payload = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: CLASS_NAME},
        "image_size": image_size,
        "label_format": "yolo",
        "label_schema": "class xc yc w h (normalized 0-1)",
    }
    (root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


def generate_split(
    split: str,
    n: int,
    cfg: GenerateConfig,
    rng: np.random.Generator,
) -> list[dict]:
    root = Path(cfg.root)
    img_dir = ensure_dir(root / "images" / split)
    lab_dir = ensure_dir(root / "labels" / split)
    records: list[dict] = []
    for i in tqdm(range(n), desc=f"generate:{split}"):
        stem = f"{i:06d}"
        img, boxes = render_scene(cfg, rng)
        img_path = img_dir / f"{stem}.png"
        lab_path = lab_dir / f"{stem}.txt"
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        write_yolo_label(lab_path, boxes, cfg.image_size, cfg.image_size)
        records.append(
            {
                "file": f"images/{split}/{stem}.png",
                "label": f"labels/{split}/{stem}.txt",
                "count": int(len(boxes)),
            }
        )
    return records


def run_generate(cfg: Config) -> dict:
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    g = cfg.generate
    root = ensure_dir(g.root)
    counts = {"train": g.n_train, "val": g.n_val, "test": g.n_test}
    manifest: dict = {
        "schema": "tube_count.v1",
        "seed": cfg.seed,
        "image_size": g.image_size,
        "label_format": "yolo",
        "class_name": CLASS_NAME,
        "splits": {},
    }
    for split in SPLITS:
        manifest["splits"][split] = generate_split(split, counts[split], g, rng)
    dump_json(root / "manifest.json", manifest)
    write_dataset_index(root, g.image_size)
    summary = {
        split: {
            "n": len(rows),
            "mean_count": float(np.mean([r["count"] for r in rows]) if rows else 0.0),
        }
        for split, rows in manifest["splits"].items()
    }
    dump_json(root / "splits.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a synthetic circular-tube dataset")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--out", type=str, default=None, help="Override generate.root")
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    if args.out:
        cfg.generate.root = args.out
    if args.n_train is not None:
        cfg.generate.n_train = args.n_train
    if args.n_val is not None:
        cfg.generate.n_val = args.n_val
    if args.n_test is not None:
        cfg.generate.n_test = args.n_test
    if args.seed is not None:
        cfg.seed = args.seed
    summary = run_generate(cfg)
    print("Wrote dataset:", summary)


if __name__ == "__main__":
    main()
