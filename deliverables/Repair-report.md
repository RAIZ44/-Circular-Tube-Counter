# Pipe Counter repair report

The local project is operational on the NVIDIA RTX 5090. The four existing
exports are the complete available image set, and their grouped v2 split is
the current training/validation/test benchmark. No additional field photos
are required for that benchmark. Accuracy on separately captured field images
remains outside the current evaluation scope.

## Completed

- Repaired the parent Python environment for the Owner account, refreshed its
  project installation and pip launcher, installed the test runner, and saved
  installed package versions in requirements-local-lock.txt.
- Migrated all 13,305 original manifest records to relative paths. Every image
  resolves locally. Original split membership, labels and model weights remain
  available for historical comparison.
- Rebuilt data/processed_v2 from the current dataset folders: 13,312 unique
  images, 1,934,146 annotated centers, 3,363 related-image groups. The largest
  group has 43 images. The rebuild recovered seven usable images absent from
  the older prepared data.
- Grouped source filenames before Roboflow suffixes, exact file hashes and
  perceptual matches transitively. Aliases from removed duplicates are retained
  so links are not lost during deduplication.
- Verified zero overlap between all split pairs for source keys, exact hashes,
  perceptual keys and group identifiers. This establishes no overlap under
  these checks; crops, renamed sources and collage components can still evade
  them. It is not proof of independent field performance.

| New split | Images |
|---|---:|
| Training | 10,650 |
| Validation | 1,332 |
| Test | 1,330 |

## Evaluation and label review

New checkpoints record the development-image identities used during training
and validation. Evaluation now reports whether known development overlap exists
or checkpoint provenance is unavailable. The original v1 checkpoint cannot
become independent of its prior training data merely by evaluating it on v2.

The original saved metrics and field_predictions directory now have status
notes explaining their limitations. The field-prediction folder contains 303
dataset examples, including 76 paths directly present in the original training
split.

The refreshed audit is in data/audit_v2. It includes 12 previews and a review
sheet. Four previews were visually inspected by the assistant; three sparse
counts were checked. One image labels a pipe side in a trench rather than a
visible open end, and two inspected images have conspicuous editing/collage
artifacts. These observations are explicitly marked as AI review, not
human-approved labels. Training labels were not silently rewritten.

The complete conflict list is data/processed_v2/annotation_conflicts.json:
140 conflicting duplicate-image annotation sets still need adjudication.
Audit and field-review creation protect existing manual review files against
accidental replacement.

## Verification

- Nine regression tests passed, including CUDA mixed-precision backpropagation
  and a verified optimizer update on the RTX 5090.
- Python source compilation passed; package consistency check found no broken
  requirements.
- A one-epoch CPU integration run completed on four training and two validation
  images, saved a checkpoint, and evaluated two separate test images. This is
  a functionality check only, not a trained-model accuracy result.
- The original saved model loaded and reproduced the nine-pipe count on the
  selected existing prediction example.
- Confirmed an NVIDIA GeForce RTX 5090 with approximately 32 GB memory using
  the NVIDIA driver diagnostic. Replaced the copied Intel-only PyTorch build
  with PyTorch 2.11.0+cu128; CUDA 12.8 now works. The initial absence of CUDA
  in PyTorch was a package mismatch, not missing GPU hardware.
- A second integration run completed two epochs on CUDA at 512 pixels, followed
  by evaluation and a successful original-model nine-pipe prediction check.
  These tiny runs verify operation only, not model accuracy.
- scripts/train_rtx5090.ps1 is configured for a full 40-epoch maximum run with
  early stopping, batch size 16, four loader workers, and test evaluation after
  training. Its syntax was checked. The full training run has not been started.

## Current dataset benchmark

Use data/processed_v2 with 10,650 training images, 1,332 validation images,
and 1,330 reserved test images. Related variants stay in the same split. The
original v1 checkpoint has seen some of the regrouped data, so a fresh v2 model
must be trained before reporting a v2 held-out test result.

The user has confirmed that these four exports are all available images.
Additional photography is not a prerequisite. The optional future field-photo
workflow remains in FIELD_BENCHMARK.md. Label spot-check findings and the 140
annotation conflicts remain visible for further review.

## Project location and recovery

Project: `C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc`

Activate the parent environment from the project directory with
`..\.venv\Scripts\Activate.ps1`, or invoke `..\.venv\Scripts\python.exe`.

Pre-edit source files, original manifests, and the former environment's launch
scripts/configuration are backed up in this task's work/before-repair directory.
Original raw images, saved model weights and historical metric files were preserved.
