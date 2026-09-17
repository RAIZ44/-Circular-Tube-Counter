# Pipe counter development candidate

**Not approved for fully automatic production use.**

Best current validation candidate: ResNet18 encoder and stride-2 center heatmap
decoder, epoch 21, 512-pixel letterbox input, float32 inference, one photo at a
time, confidence threshold 0.325, 3x3 local maxima, image-content masking, and
a maximum of 2,000 peaks. RGB normalization and counting code are included.

On 1,332 validation images: 28.15% exactly correct counts, 41.82% within one,
mean absolute error 9.33 pipes. Model/threshold selection used this validation
set. The reserved test set remains unused. These are not production accuracy
estimates. Dense scenes, collages, label conflicts, and nested-pipe rules remain
unresolved. This package preserves the current best work for further development.

## Run locally

Use a Python environment with the included pinned dependencies; the local lock
is for this Windows/CUDA installation, not an Azure deployment specification.
Install this folder as a package using `python -m pip install -e .` after the
dependencies are available. ResNet inference requires torchvision. No network
download is needed to load candidate.pt once dependencies are installed.

`python scripts/predict.py --checkpoint candidate.pt --input regression_examples --output-dir predictions --device cuda --disable-geometry`

Use `--device cpu` for CPU execution. Cross-device and other GPU runtimes require
parity testing; exact numerical equivalence is not assumed. Preserve the saved
precision, preprocessing, threshold and single-photo inference configuration.

`validation_report.json` records full errors and source/density slices.
`regression_examples.json` records expected counts for twelve validation images,
which are regression fixtures, not independent evaluation examples.
`training_provenance.json` records training dependencies, GPU, manifest hashes,
and the official ImageNet initialization source. The checkpoint records its
parent hash and validation calibration provenance. Original training artifacts
remain in the project. No Azure deployment or test-set evaluation was performed.
