# Circular Tube Counter

Count circular tubes (pipe/tube cross-sections, rings viewed roughly top-down)
with a small **CenterNet-style** PyTorch detector. Count = number of detections
above a confidence threshold calibrated on the validation split.

The repo ships a full **generate → train → evaluate → predict** loop, a YOLO
data contract so real photos can replace synthetics, and two baselines
(always-predict mean count, OpenCV Hough circles).

## Install

Python 3.10+ . CPU works; CUDA is used automatically when available.

```bash
# CPU torch (skip if you already have a GPU build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install -e .
```

`pip install -e .` alone also pulls dependencies from `pyproject.toml` if torch
is already installed (or if you are fine with the default PyPI torch wheel).

## One path: generate → train → evaluate

All commands below are from the repo root.

```bash
python -m tube_count.generate --config configs/default.yaml
python -m tube_count.train --config configs/default.yaml
python -m tube_count.evaluate --config configs/default.yaml --checkpoint artifacts/checkpoints/best.pt
```

Equivalent: `tube-count generate|train|evaluate`.

Default synthetic set: **360 train / 72 val / 80 test** images at 256×256,
0–16 tubes each, with overlap, clutter, lighting, and noise.

### Expected runtime (measured on this 4-core CPU VM)

| Step | CPU (4 cores) | GPU (typical 8–12 GB) |
| --- | --- | --- |
| Generate 512 images | ~7 s | same |
| Train 20 epochs (256×256, batch 8) | ~4.3 min (~13 s/epoch) | ~1–2 min |
| Evaluate + baselines | ~4 s | ~1–2 s |
| Predict, one image | ~1 s including load | faster |

### Verified test results (seed 42, this VM)

Held-out **80** synthetic test images, checkpoint `artifacts/checkpoints/best.pt` (epoch 10, val MAE 0.306, conf 0.3):

| Method | MAE | RMSE | Exact-match | F1 @ IoU 0.5 |
| --- | --- | --- | --- | --- |
| **TubeNet (ours)** | **0.463** | **0.814** | **62.5%** | **0.869** |
| Always predict mean train count (8) | 4.363 | 4.862 | 5.0% | n/a |
| OpenCV HoughCircles | 7.513 | 9.308 | 3.8% | 0.474 |

Overcount 12.5% / undercount 25.0%. Full dump: [`artifacts/metrics.json`](artifacts/metrics.json), write-up [`artifacts/REPORT.md`](artifacts/REPORT.md), overlays in [`artifacts/overlays/`](artifacts/overlays/). Checkpoint SHA256 is in [`artifacts/checkpoints/best.meta.json`](artifacts/checkpoints/best.meta.json).

The image dataset is **not** committed (regenerate with the same config/seed). The checkpoint **is** committed (~3 MB).

## Predict

```bash
python -m tube_count.predict --input path/to/image.png --checkpoint artifacts/checkpoints/best.pt
python -m tube_count.predict --input path/to/folder --checkpoint artifacts/checkpoints/best.pt --out artifacts/pred
```

Prints a count per image. With `--out`, also writes annotated PNGs plus
`predictions.csv` / `predictions.json`.

## Label schema / real data

Synthetic images are a stand-in. Train and eval only read YOLO files:

- `data/images/{train,val,test}/<stem>.png`
- `data/labels/{train,val,test}/<stem>.txt`
- each line: `0 <xc> <yc> <w> <h>` (normalized 0–1)
- empty file = count 0

Full contract: [`data/README.md`](data/README.md). To use real photos, write that
layout and skip `generate`. Keep `generate.image_size` in the config equal to the
training resize.

## What gets trained

`TubeNet` is a tiny stride-4 encoder/decoder (~0.5M params):

- **heatmap** head: Gaussian peaks at tube centers (focal loss)
- **radius** head: outer radius at those peaks (L1)
- decode: 3×3 max-pool NMS → boxes → **count = detections ≥ threshold**
- threshold is swept on val MAE and stored in the checkpoint

Classical Hough circles is a **baseline only**, not the primary model.

## Artifacts

After a successful eval:

| Path | Contents |
| --- | --- |
| `artifacts/checkpoints/best.pt` | Best val-MAE weights + calibrated threshold |
| `artifacts/checkpoints/best.meta.json` | Epoch, MAE, SHA256 digest |
| `artifacts/logs/train.jsonl` | Per-epoch loss, LR, val metrics |
| `artifacts/metrics.json` | Test MAE/RMSE/exact-match, P/R/F1, baselines |
| `artifacts/REPORT.md` | Human-readable summary |
| `artifacts/overlays/*.png` | GT (green ellipse) vs pred (red box) |

Reproducibility: `seed: 42` in `configs/default.yaml` seeds Python, NumPy, and
Torch. Checkpoint SHA256 is recorded next to the metrics.

## Tests

```bash
python -m pytest -q
```
