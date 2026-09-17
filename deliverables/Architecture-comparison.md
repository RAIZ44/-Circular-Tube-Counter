# Architecture comparison and accuracy target

The working target is **at least 95% of photos counted exactly right**. The
current best exact-count validation result is 29.88%; this is a target to work
toward, not a promise that the existing data can establish production accuracy.
The reserved test remains unused.

| Candidate | Status | Reason to evaluate |
|---|---|---|
| Higher-resolution Faster R-CNN | Completed; 29.58% exact, no promotion | Larger input reduced dense-scene errors; adaptation may recover exact counts |
| FCOS ResNet50 FPN | CPU and dense GPU checks passed; full training not yet started | Different, single-stage detection method; compare count errors on the same split |
| RF-DETR | Research candidate | Transformer-based detection; must first address dense-scene object limits |
| YOLO26 / small-object variant | Researched alternative | Another detector family; requires separate training and deployment review |

FCOS has 2,000 output slots and a larger candidate budget for this experiment.
Its official pretrained weights loaded successfully; 28 CPU tests passed,
including exact offline reload checks. A CPU training step with positive and
empty examples passed. Full-resolution GPU preflight passed on the RTX 5090, including two images
with 1,523 labels each. The comparison is capped at 12 epochs with early
stopping; it has not yet demonstrated an accuracy improvement.

RF-DETR defaults to 300 queries for core models. Its documentation requires
increasing both query and output counts to handle more objects, followed by
fine-tuning. This must be addressed for photos containing hundreds or thousands
of pipes. [Official RF-DETR FAQ](https://github.com/roboflow/rf-detr/blob/develop/docs/faq.md)

FCOS is a one-stage, anchor-free detector.
[Official Torchvision FCOS documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fcos_resnet50_fpn.html)

YOLO26 supplies a small-object P2 architecture as YAML; pretrained transfer and
dense output limits need validation before a comparison.
[Official YOLO26 documentation](https://docs.ultralytics.com/models/yolo26)

All comparisons retain grouped train/validation membership. Original images and
labels are preserved. Architectural improvements must be measured alongside
label quality and performance on the intended side-by-side setup.
