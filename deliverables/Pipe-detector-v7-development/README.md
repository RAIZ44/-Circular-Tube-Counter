# Pipe detector v7 — development package

**Not approved for fully automatic production.** This package preserves the
latest lower-error detector for local development and later Azure integration.
The intended use is side-by-side pipe ends; nesting is outside scope.

On the mixed 1,332-image validation set: 397 exactly correct counts (29.80%),
45.95% within one pipe, average absolute error 8.35 pipes and RMSE 29.72.
Model and threshold selection used this validation set. These figures do not
establish performance on actual operating photos. The reserved test is unused.
The previous detector has one more exactly correct photo, but average error
10.40; it remains preserved separately. Training stopped after five epochs;
this is the best checkpoint from epoch two.

## Run

Use the existing project Python environment, or an environment with the versions
in requirements-local-lock.txt. That lock describes the local Windows/CUDA
installation, not a portable Azure deployment specification. This package uses
its bundled source directly and needs no editable installation. Once dependencies
exist, loading the checkpoint does not download pretrained weights.

From this directory:

```text
python scripts/predict_detector.py --checkpoint candidate.pt --input regression_examples --output-dir predictions --device cuda
```

Set input to a photo or photo directory. Use --device cpu for CPU execution.
CPU execution is functional but is not count-equivalent to the validated CUDA
reference: 15 of 16 fixture counts matched; one was 51 on CPU versus 50 on CUDA.
The output contains a CSV count summary and a JSON file for every photo with
boxes and scores. The reported detection-limit flag means the internal 2,000-box
limit was reached; the model cannot establish that a capped count is complete.

## Fixed inference behavior

Faster R-CNN ResNet50 FPN V2, float32 tensors, one photo at a time, RGB values in [0,1].
The supplied CUDA runner explicitly enables cuDNN TF32 and disables matrix-multiply
TF32 to match the recorded validation runtime. The tensor type alone does not
fully specify arithmetic. Turning cuDNN TF32 off changed the borderline fixture
from 50 to 51, matching CPU. This setting is pinned for reproducibility, not as
evidence that either device's count is correct. Other GPU models still need parity checks.
The model applies ImageNet normalization and resizes the longest side to 640
while preserving aspect ratio. Confidence threshold is saved near 0.43;
duplicate suppression uses IoU 0.2. Preserve the saved values rather than
rounding or recalibrating at deployment. Source and preprocessing are bundled.
Annotations' geometry corrections are included for provenance; they do not
rotate or otherwise change prediction images.

## Evidence and limitations

The regression examples are 16 selected validation fixtures, covering count
ranges, an empty image and known annotation-geometry cases. Expected predictions
are saved from the validation run. They test reproducibility, not accuracy.
Verification results are in reload_verification.json when present.
All 16 reference counts reproduced on the RTX 5090. Warm model forward time had
a median of 37.94 ms across these fixtures, excluding image decoding, tensor
creation, startup and file writing; this is not an end-to-end or Azure guarantee.

Known limitations include inconsistent labels at cropped image boundaries,
missing or duplicate labels, duplicate detections, small-end misses and dense
scenes. Existing grouped splits reduce known related-image overlap but cannot
rule out shared collage components or renamed scenes. No accuracy claim is made
for an automatically filtered side-by-side subset. No Azure deployment has
occurred; other runtimes or accelerators require parity and performance checks.

candidate.pt retains development provenance and geometry metadata; optimizer
state was removed without changing weights or inference settings. Original
training checkpoints remain preserved. SHA256SUMS.json records package hashes.
