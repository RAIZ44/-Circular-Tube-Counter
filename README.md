# Circular Tube Counter

Pipe-end counting research, source code, all four supplied COCO dataset exports,
frozen data splits, experiment checkpoints, validation results, and audit reports.

**Development status: not production ready.** The target is at least 95% of
photos counted exactly correctly. The best measured validation exact-count
accuracy is **29.88% (398 of 1,332 photos)**. This is different from object
detection mAP or aggregate pipe-count error. The reserved 1,330-image test
split has not been evaluated for candidate selection.

## Download

Large images, checkpoints, and manifests use Git LFS. Install Git LFS before
cloning, then run:

```sh
git lfs install
git clone https://github.com/RAIZ44/-Circular-Tube-Counter.git
cd -- -Circular-Tube-Counter
git lfs pull
```

The GitHub source ZIP is not a substitute for a verified LFS checkout.

## Repository layout

| Folder | Contents |
| --- | --- |
| `pipe-center-counter-poc/` | Python package, tests, training/inference scripts, manifests, experiment history and checkpoints |
| `pipe-datasets/` | All four original COCO exports with images and annotations |
| `deliverables/` | Packaged development candidates, accuracy reports, visual label/error audits |
| `research/` | Supporting experiment and audit scripts, diagnostic results, preserved source before repairs |

Keep the project and dataset folders beside each other: the frozen manifests
resolve paths relative to their own location. Original export split names are
preserved for provenance; **use `data/processed_v2` for experiments**. It has
10,650 train, 1,332 validation, and 1,330 reserved test images, grouped to keep
known related images together.

Installed environments, downloadable model caches, Python caches, temporary
test folders, and live process locks are excluded. Checkpoints and historical
logs are included. Historical absolute paths and process IDs describe the
original machine and are not instructions to resume a live process.

## Current results

All results below use the same frozen validation set; thresholds were selected
on validation. They are development scores, not an independent production test.

| Model/checkpoint | Exact count | Mean absolute count error |
| --- | ---: | ---: |
| ResNet18 heatmap, float32 | 28.15% | 9.33 |
| Faster R-CNN v6, calibrated | **29.88%** | 10.40 |
| Faster R-CNN v7 | 29.80% | 8.35 |
| Faster R-CNN v8 at 960px, selected checkpoint | 29.58% | 9.63 |

V8 completed four epochs and did not improve exact accuracy. Its best and last
checkpoints are preserved. FCOS is implemented with official pretrained
weights, 28 passing CPU checks, and a successful RTX 5090 dense-batch training
preflight. **Full FCOS training has not started in this snapshot**: the attempted
background launch was blocked by the local approval service's usage limit.
RF-DETR and YOLO are researched alternatives, not evaluated models.

See [acceptance criteria](pipe-center-counter-poc/ACCEPTANCE_CRITERIA.json),
[readiness notes](pipe-center-counter-poc/PRODUCTION_READINESS.md), and
[architecture comparison](deliverables/Architecture-comparison.md).

## Local use

The original workstation uses an NVIDIA RTX 5090, Python 3.14,
PyTorch 2.11.0+cu128, and torchvision 0.26.0+cu128. Exact environment versions
are recorded in `pipe-center-counter-poc/requirements-local-lock.txt`.

```sh
cd pipe-center-counter-poc
python -m venv ../.venv
# Activate the environment for your shell, then:
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
python scripts/predict_detector.py --checkpoint runs/pipe_detector_v7_finetune/best.pt --input path/to/photo.jpg --output-dir predictions --device cuda
```

For a self-contained development inference bundle, see
[`deliverables/Pipe-detector-v7-development`](deliverables/Pipe-detector-v7-development).
CPU/GPU counts can differ near score thresholds; the CUDA reference math
settings and regression fixtures are recorded in that bundle.

## Data and evaluation limitations

The intended deployment uses pipes arranged side by side; nested pipes are
outside the stated scope. Existing annotations have disagreements about
rectangular tubes, cropped ends, and duplicate boxes. No unanswered label
policy was silently imposed. Geometry corrections and a staged duplicate-box
revision are versioned; current architecture comparisons retain the original
frozen labels. More training alone does not establish 95% accuracy.

Original dataset attribution and license notices are retained in the export
folders. This snapshot does not introduce a blanket license over third-party
images or weights. Azure deployment is not included.

`SNAPSHOT_INVENTORY.json` records source-file hashes before publication-only
documentation and ignore-file changes. `PUBLICATION_SHA256.json` records final
published content hashes for integrity verification.
