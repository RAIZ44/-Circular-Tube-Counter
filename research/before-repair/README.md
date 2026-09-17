# Pipe Center Counter POC

Point-supervised pipe-end counting from Roboflow COCO object-detection exports.

The project converts each `pipe`/`tube` bounding box to one center point, trains a
small CNN to predict a stride-2 heatmap of pipe centers, extracts one local
maximum per pipe with OpenCV/PyTorch, and flags suspicious gaps in the staggered
row layout.

## What the POC does

1. Recursively discovers every COCO JSON file below a dataset root.
2. Treats the `pipe` and `tube` categories as one `pipe_end` target.
3. Deduplicates identical images by SHA-256. If duplicate copies have different
   annotations, the copy with the most valid boxes is retained and reported.
4. Groups matching perceptual hashes into the same deterministic
   train/validation/test split, reducing leakage from resized or re-encoded
   copies without automatically deleting them.
5. Converts bounding boxes into Gaussian center-point heatmaps at training time.
6. Tunes the peak-confidence threshold on validation data.
7. Reports exact-count accuracy, within-one accuracy, and mean absolute count
   error on the test set.
8. Produces annotated predictions without silently adding geometry-inferred
   pipes to the detected count.

## Expected dataset layout

The exact folder names do not matter. Point `--data-root` at the common parent:

```text
pipe-datasets/
├── first_train/
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   └── *.jpg
│   ├── valid/
│   └── test/
├── second_train/
├── third_train/
└── fourth_train/
```

The preparation script searches recursively, so Roboflow's generated folder
names are supported as-is.

## 1. Set up on Windows

Open PowerShell in the extracted project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

If PowerShell blocks activation, run this once in that terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Intel Arc GPU and Intel AI Boost NPU

For a laptop with Intel Arc graphics, replace the default PyTorch package with
the official XPU build. Keep the virtual environment active and run:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install -e .
python scripts\check_accelerators.py
```

For training, the diagnostic should report:

```text
"pytorch_xpu_available": true
"pytorch_xpu_device": "Intel(R) Arc(TM) 130V GPU"
```

The Arc GPU accelerates training, validation, and development predictions. The
POC's `--device auto` now selects XPU automatically, or it can be required with
`--device xpu`. FP16 autocast is enabled on XPU, while gradient scaling is
intentionally disabled for Intel client Arc compatibility.

The NPU is intended for power-efficient deployed inference after training, not
PyTorch training. To make it visible to the optional OpenVINO runtime:

```powershell
pip install -e ".[intel-inference]"
python scripts\check_accelerators.py
```

When the driver/runtime is ready, `openvino_devices` should include `NPU` (and
usually `CPU` and `GPU`). Exporting and benchmarking the trained checkpoint on
all three OpenVINO devices is a later deployment step; do not delay initial
training for it.

Intel integrated graphics use shared system memory. A value such as 16 GB in
Task Manager is normally an upper shared-memory allowance, not 16 GB of
dedicated VRAM. Start with batch size 2 or 4 and reduce it if Windows reports an
out-of-memory error.

If you have an NVIDIA GPU instead, install the matching CUDA-enabled PyTorch
build before `pip install -e .`. Confirm availability with:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Prepare and inspect all four datasets

Replace the example path with the parent folder on your computer:

```powershell
python scripts\prepare_data.py `
  --data-root "C:\Users\Owner\Downloads\pipe-datasets" `
  --output-dir "data\processed" `
  --classes pipe tube
```

Outputs:

```text
data/processed/
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── dataset_report.json
└── previews/
```

Open the preview images before training. Every visible pipe end should have one
green center dot. If boxes label whole side-view pipes instead of visible ends,
those images are not suitable for this center-point formulation and should be
removed or relabeled.

## 3. Audit density and annotation quality

The four exports contain unusually dense images, so compare stride-2 and
stride-4 center collisions and generate sparse/typical/dense/extreme previews:

```powershell
python scripts\audit_data.py `
  --manifest-dir "data\processed" `
  --output-dir "data\audit" `
  --image-size 512
```

Review `data\audit\previews`, then optionally record manual visible counts and
missing/false labels in `data\audit\manual_label_audit.csv`. A visible pipe with
no dot is a missing ground-truth label; the preview itself is not making model
predictions.

## 4. Run a quick smoke training

Start small to confirm the pipeline works:

```powershell
python scripts\train.py `
  --manifest-dir "data\processed" `
  --output-dir "runs\smoke" `
  --image-size 384 `
  --batch-size 2 `
  --epochs 2
```

## 5. Train the POC

For the Intel Arc GPU (or an NVIDIA GPU selected automatically):

```powershell
python scripts\train.py `
  --manifest-dir "data\processed" `
  --output-dir "runs\pipe_center_v1" `
  --image-size 512 `
  --batch-size 4 `
  --epochs 40 `
  --patience 8 `
  --device auto
```

If memory runs out, reduce `--batch-size` to `4` or `2`. CPU training works but
will be considerably slower; use `--image-size 384` for the first experiment.

The best checkpoint is saved as:

```text
runs/pipe_center_v1/best.pt
```

## 6. Evaluate exact counting performance

```powershell
python scripts\evaluate.py `
  --checkpoint "runs\pipe_center_v1\best.pt" `
  --manifest "data\processed\test.jsonl" `
  --output "runs\pipe_center_v1\test_metrics.json"
```

The main metric is `exact_count_accuracy`, not bounding-box mAP.

## 7. Count pipes in new images

For one image:

```powershell
python scripts\predict.py `
  --checkpoint "runs\pipe_center_v1\best.pt" `
  --input "C:\path\to\truck_photo.jpg" `
  --output-dir "runs\predictions"
```

For every image in a folder:

```powershell
python scripts\predict.py `
  --checkpoint "runs\pipe_center_v1\best.pt" `
  --input "C:\path\to\truck_photos" `
  --output-dir "runs\predictions"
```

Prediction images contain:

- green dots: detected pipe centers;
- yellow crosses: suspicious lattice gaps for operator review;
- `Detected`: the model count;
- `Possible gaps`: geometry warnings only, never silently added to the count.

## Useful controls

```text
prepare_data.py --duplicate-policy max_annotations|first
prepare_data.py --no-perceptual-grouping
train.py        --device auto|xpu|cuda|cpu
train.py        --num-workers 0
predict.py      --threshold 0.35
predict.py      --disable-geometry
```

On Windows, use `--num-workers 0` if DataLoader workers cause an error.

## Important POC limitations

- Exact-file hashing removes byte-identical duplicates. A lightweight
  perceptual hash keeps likely resized/re-encoded copies in one split and lists
  them for review, but it is not a complete near-duplicate detector.
- Center labels inherited from bounding boxes are only as good as the original
  annotations.
- Geometry validation assumes mostly front-facing, staggered rows and is a
  warning system, not a source of automatic count corrections.
- A production benchmark should include held-out photographs from the actual
  Sonoco camera position, lighting, pipe dimensions, pallets, and trucks.
- The exported metadata says CC BY 4.0, which requires attribution. Verify the
  provenance and permitted use of every third-party/stock image before using
  this dataset or derived weights beyond a POC; production training should
  prioritize internally captured, approved images.
