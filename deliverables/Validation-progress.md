# Pipe counter progress

**Development continues; the model is not ready for fully automatic production.**

All results below use the same 1,332 validation photos. Thresholds were tuned on
validation, so these are development scores. The reserved test remains unused.

| Preserved candidate | Exactly correct photos | Within one pipe | Average absolute error |
|---|---:|---:|---:|
| Center model, full precision | 28.15% | 41.82% | 9.33 pipes |
| Box detector, calibrated | 29.88% (398/1,332) | 46.92% | 10.40 pipes |
| Box detector, latest fine-tune | 29.80% (397/1,332) | 45.95% | 8.35 pipes |

The latest run finished normally after five epochs; epoch two was best. Average
error improved about 20% against the previous detector, with one fewer exactly
correct photo. Both checkpoints remain preserved. No additional full training
run is active; the current work is error and annotation review.

The target is side-by-side pipes, with nesting outside the intended use. That
narrower scope does not establish an accuracy figure for actual operating photos.
The full mixed validation reference remains unchanged.

Ten selected error images show inconsistent treatment of cropped ends, missing
labels, duplicate detections and misses on small distant ends. These are visual
diagnostics, not a representative sample. No labels were silently corrected.
See [the illustrated review](Side-by-side-error-review/index.html).

Next work: audit exact duplicate training annotations and establish a consistent
rule for partial ends, then version any verified corrections before another
bounded experiment. Avoid further blind epoch extensions. Monitoring remains
configured; the reserved test and Azure deployment are untouched.

The latest detector is now preserved as a [development package](Pipe-detector-v7-development/README.md).
All 16 saved counts reproduced on the RTX 5090. CPU matched 15 of 16; one
near-threshold detection changed the count by one, so CPU/GPU parity is not
approved. The CUDA runner pins the audited arithmetic settings.

The [training-label audit](Training-duplicate-audit/README.md) traced all 409
exact duplicate boxes in 49 training records to the original exports. A separate
training-only revision is staged, with raw files and frozen splits preserved.
It has not been used in training and does not establish an accuracy improvement.

Further review reconstructed 113 conflicting training-image annotations from
313 source copies. Some differences reflect counting rules: one photo is labeled
as 7 round pipes or 68 ends including rectangular tubes. Others differ on partial
ends or background stacks. [View the examples](Training-conflict-audit/index.html).
Questions about shapes and cropped ends are pending before changing these labels.
Training is currently paused; preserved model accuracy is unchanged.

Work resumed at the user's request without changing ambiguous labels. A 960px
comparison reduced average error from 8.35 to 6.33 pipes and error on 501+ pipe
photos from 47.3 to 26.3, but exact counts fell from 29.80% to 27.40%.
The existing model remains preserved. One bounded resolution-adaptation run is
planned, capped at six epochs with early stopping, to test whether training at
that resolution can recover exact-count performance. The reserved test remains unused.

The bounded 960px adaptation is now running on the RTX 5090, with a maximum of six epochs and early stopping. Monitoring remains active.

The user set a 95%+ target and authorized other architectures. The working
metric is exactly correct photos. FCOS is implemented and CPU-tested, queued
after the active high-resolution run; RF-DETR and YOLO are additional researched
options. See [the architecture comparison](Architecture-comparison.md).


## V8 complete; FCOS GPU preflight passed (2026-09-16T23:19:01.981522+00:00)

V8 stopped normally after four epochs, best epoch two: 394/1332 exactly
correct (29.5796%), MAE 9.6321. V7 remains 397/1332 (29.8048%), MAE
8.3529; v6 calibrated remains best exact at 398/1332 (29.8799%). V8 last
epoch had MAE 7.9444 and 392 exact photos; preserve both checkpoints without
promotion or extension. Density comparison: runs/supervision/detector_v8_comparison.json.

FCOS 960px CUDA smoke passed dense batches with 1523 labels each and a
10-label/empty batch, including optimizer steps and float32 inference. Peak
GPU allocation 2.13 GiB. This establishes execution feasibility, not accuracy.
Proceed with the prepared fresh-pretrained FCOS run: batch two, LR 1e-4,
maximum 12 epochs, patience four, NMS .3, unchanged training/validation labels.
95% exact-count target is authoritative; test remains reserved.
