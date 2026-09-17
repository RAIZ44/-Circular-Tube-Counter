# Pipe counter: supervised model development

User objective: deliver a model with strong enough accuracy for fully automatic
pipe counting. The model will later be hosted on Azure; cloud deployment is not
part of the present request. The four supplied exports are all available images.

Current status: NOT production-ready. Baseline finished after 24 epochs;
validation diagnosis completed; runs/pipe_center_v3_768_exact is now training.
Started 2026-09-16 03:24 UTC; CUDA progress verified. See status.json for current PIDs and log.
Monitor: pipe-counter-production-readiness, every 30 minutes in the current task.

## Acceptance criteria and scope

Provisional engineering target: at least 99% exactly correct counts on the
reserved test images, with within-one accuracy, MAE, signed bias, error percentiles,
and binomial confidence intervals also reported. This is a target, not a promised
result or an established business tolerance. Do not lower it to declare success.
Break results down by count/density, source, image quality, and known failure types.
Model confidence alone is not evidence of count correctness.

The supplied export benchmark can establish held-out dataset performance only.
Do not imply proven performance on unseen operational Azure inputs. Describe the
supported image conditions and any remaining validation gap explicitly. Do not
label automatic operation production-ready without suitable evidence. If current
data cannot establish it, deliver the best validated candidate and concrete blockers.

## Frozen data and evaluation policy

- data/processed_v2: 10,650 training, 1,332 validation, 1,330 reserved test images.
- Source-name, exact-hash and perceptual matches are grouped transitively. Known
  overlap is zero. Crops, renamed images and collage components can evade these checks.
- The 140 annotation conflicts and AI spot-check notes in data/audit_v2 require
  adjudication. Do not treat AI observations as human-approved labels or blindly
  replace labels with model predictions. Preserve raw data and record corrections.
- Use validation for threshold tuning and experiment selection. Do not inspect
  test errors to choose repeated improvements and continue calling it untouched.
- The first supervised job trains and validates only. Review that evidence and
  freeze a candidate before a test evaluation. Log every test-set access and any
  later reuse. The v1 checkpoint has prior exposure to some v2 groups.

## Active job and supervision

Use ..\.venv\Scripts\python.exe. CUDA build: PyTorch 2.11.0+cu128 on an RTX 5090.
scripts/supervised_train.py launches a fresh runs/pipe_center_v2 experiment:
512 pixels, batch 16, 4 workers, up to 40 epochs, patience 8, seed 42.
It records source snapshots, manifest hashes, versions, PID, logs and phase.

Read runs/supervision/status.json, training.log and runs/pipe_center_v2/history.csv.
For subsequent runs use the log and run_dir fields in status.json; the original
runs/supervision/training.log is historical. Experiments are recorded in
runs/supervision/experiments.json. No concurrent GPU training is permitted.
Check the process ID and command before deciding a job is stalled. Do not start
another training job while one is running. job.lock intentionally persists after
completion/failure; inspect state and existing artifacts before explicitly
archiving it and choosing a new run directory. Never overwrite prior checkpoints.

The monitor may repair failures, add meaningful tests, investigate label issues,
and perform bounded local experiments. It must record hypotheses and validation
results. Stop repeating approaches that are not improving results; explain a
data or modeling limitation instead. Do not edit source used by an active run;
prepare later changes separately, or wait until the run finishes.

Notify only for completed evaluations, material improvements, failures, important
milestones or required decisions. Stay quiet during normal unchanged progress.
No paid cloud jobs, external publishing, live deployment or driver changes are
authorized by the monitoring request.

## Work queue

1. Confirm CUDA training makes progress; capture the first validation baseline.
2. Align validation counting with prediction counting: currently validation can
   include peaks on letterbox padding and does not apply prediction's peak cap.
   Correct and regression-test this before final candidate selection/reporting.
3. Produce validation per-image errors and density/source breakdowns. Review
   worst failures and annotation conflicts. Test evidence-based improvements.
4. Assess threshold selection against the exact-count objective (the current
   baseline optimizes MAE first), resolution/crop handling, and dense-stack errors.
5. Freeze the selected model and thresholds; evaluate reserved test data with
   confidence intervals and transparent label-quality limitations.
6. Deliver weights, inference code, environment lock, preprocessing/threshold
   configuration, regression examples, reproducibility metadata and a model card.
   Verify a clean reload and consistent outputs. Measure inference latency on
   known hardware; Azure performance requires a specified Azure runtime later.
7. Record which readiness gates pass, fail or lack evidence. Pause the monitor
   once accepted requirements are demonstrably met, or explain a necessary blocker
   once and await input rather than repeatedly retraining without a path forward.

## Validation investigation, 2026-09-16 03:11 UTC follow-up

Completed: baseline training (24 epochs, best epoch 16), content-boundary masking
and peak-cap consistency between validation/prediction, per-image error reports,
source/density breakdowns, exact-count threshold selection, and 16 passing tests.
The mismatch described in work item 2 is now corrected.

Validation-only comparisons on 1,332 images:

| Candidate | Exact counts | MAE | Threshold |
|---|---:|---:|---:|
| Original selected checkpoint report | 17.34% | 10.05 | 0.250 |
| Same checkpoint, fine MAE threshold search | 18.62% | 9.92 | 0.260 |
| Same checkpoint, exact-count threshold search | 20.20% | 10.90 | 0.290 |
| Last checkpoint, exact-count threshold search | 19.44% | 11.68 | 0.295 |
| 768 input without adaptation, exact-count threshold | 11.56% | 28.27 | 0.430 |
| Context tiles, exact-count threshold | 10.21% | 12.21 | 0.250 |

Do not promote larger-input or tile-only inference: neither improved exact
accuracy. Retain the original 512 baseline. Direct 768 inference did reduce RMSE
at its MAE-optimal threshold (36.85 to 25.50) but needs learned scale adaptation.
Next bounded experiment: initialize from v2 best, train at 768 pixels, batch 8,
learning rate 0.0001, up to 15 epochs, patience 5; select threshold/checkpoint by
exact-count accuracy with MAE as tie-breaker. Hypothesis: adapting features to
the higher resolution can recover dense/small ends without the unadapted scale
penalty. Compare on validation only against 20.20% exact / 10.90 MAE at 512.
Reject it if it fails to improve; do not endlessly increase resolution.

Target audit: ideal heatmap targets fail to reproduce counts on 65/1,332 validation
images at 512 and 55 at 768. This diagnoses target collisions/boundaries, not a
universal accuracy ceiling. Exact duplicated boxes occur in 49 training images
(409 excess labels) and 10 validation images (10 excess labels). No labels were
silently changed. Worst failures include a 700x100 repeated square-pipe panorama
with 987 labels and nested-pipe imagery; collage and target-definition issues need
further review. Dense counts over 500 are only 5.2% exact in the 512 diagnostic.

Reports: runs/validation_diagnostics/v2_best, v2_last and v2_best_tiles.
The reserved test set has not been evaluated. Current scores are validation-tuned
and must not be presented as held-out test or production accuracy.

## Second supervised review, 2026-09-16 04:02 UTC

The 768-pixel fine-tune completed all 15 epochs normally. Best epoch 10 reached
16.97% exact counts, 30.41% within one, MAE 13.26 at threshold 0.255. Rejected:
the preserved 512 baseline remains better at 20.20% exact counts and MAE 10.90.

The next bounded run, pipe_center_v4_resnet18_exact, is training on the RTX 5090:
ImageNet-pretrained ResNet18 encoder plus a stride-2 center heatmap decoder,
512 pixels, batch 16, up to 30 epochs, patience 6, decoder learning rate 0.0003,
encoder learning rate 0.00003. Select checkpoint and threshold by validation
exact-count accuracy, with MAE as tie-breaker. Same manifests and existing labels.
Official pretrained weights are cached and SHA256 recorded in run provenance.
Checkpoint reload uses local weights and does not download pretrained weights.
Eighteen tests passed, and a short CUDA training run passed before full launch.

Audit reports identify 140 identical-image annotation conflict sets, of which
100 have differing counts, including 7 versus 68 and 360 versus 531. These
conflicts span raw exports; they are not a new reserved-test performance result.
The correct definition for nested pipes (every visible end versus outer pipes)
was asked of the user and remains unresolved. Do not infer approval from silence.
Do not silently change labels or remove hard examples to improve scores.

If pretrained features fail to materially improve validation performance,
prioritize label adjudication and target representation instead of repeated
speculative training. Neither these validation scores nor more epochs establish
production readiness. The reserved test set remains unevaluated.

## Precision repair and candidate review, 2026-09-16T08:18:26.109223+00:00

The pretrained run completed normally after 27 epochs, best epoch 21. Its legacy
mixed-precision validation result was 23.42% exact. A diagnostic exposed FP16
sigmoid rounding that turns distinct adjacent scores into equal plateaus and
counts both peaks. Float32 sigmoid helps; full float32 model inference improves
exact counts further. Precision is now explicit in checkpoints and shared by
prediction/evaluation. Old checkpoints retain legacy behavior unless explicitly
recalibrated, so saved historic results are not silently redefined. New training
uses float32 validation at batch size one while retaining AMP for training.

Fair float32, single-photo validation comparisons, thresholds tuned on validation:

| Candidate | Exact counts | MAE | Threshold |
|---|---:|---:|---:|
| Earlier v2 best | 23.95% | 11.36 | 0.295 |
| Pretrained v4 epoch 21 | 28.15% | 9.33 | 0.325 |
| Pretrained v4 epoch 27 | 27.33% | 8.67 | 0.295 |

Retain epoch 21 for the exact-count objective. Larger 768 input without adaptation
again regressed (19.89% exact), so it was rejected. Batch-size changes can alter
borderline counts slightly: batch 8 gave 28.08%, batch 1 gave 28.15%. The current
candidate fixes single-photo inference. Azure/runtime changes need parity checks.

Twenty tests passed, including a regression test for sigmoid-created plateaus.
A new tiny CUDA training integration run passed. A portable development candidate
was saved with source, dependencies, calibration provenance, validation report,
and twelve regression images. Packaged-source reload reproduced all twelve counts;
CPU and CUDA matched on all twelve. Warm local preprocessing/model/peak extraction
on these fixtures: GPU median 6.12ms, p95 7.34ms; CPU median 62.63ms, p95 63.82ms.
This excludes disk decoding, model startup and rendering and is not an Azure SLA.

Accuracy still fails the provisional production gate: 375/1332 exact counts,
41.82% within one, 95th-percentile absolute error 43.0,
worst error 646 pipes. Images without a recorded identical-image
label conflict are only 28.45% exact (1318 images);
known label conflicts do not explain the overall shortfall. Dense scenes with
501+ labels are only 2.60% exact. No production approval is warranted.

Created a visual review sheet of ten training/validation label conflicts (92
different-count conflict sets belong to these development splits), plus the
nested-pipe example. No reserved-test images are included in this sheet. No labels
were changed. Await the already-requested nested-pipe counting rule before label
adjudication and label-dependent retraining; do not infer a rule from silence.
The 30-minute monitor remains active but should stay quiet while this decision is
unchanged, and should not repeatedly launch speculative jobs. No full training job
is active. Candidate: runs/pipe_center_v4_float32/candidate.pt. Test remains unused;
all frozen manifest hashes match the completed run's provenance.

## Side-by-side operating scope and resumed work, 2026-09-16T11:56:04.026407+00:00

The user clarified that nesting will not be a problem and the pipes will be beside
each other. Treat nested pipes as outside the intended operating case; no choice
between inner-versus-outer counts is required to continue. OPERATING_SCOPE.json
records the original clarification and interpretation. Remove the previous
count-definition blocker. Do not keep asking the nested-pipe question. Other label
conflicts still exist; no labels are silently changed or guessed from this reply.

Next bounded experiment: pipe_center_v5_crop_exact, initialized from the preserved
28.15% exact-count full-precision candidate. Train at 512px, batch 16, LR 0.00005,
up to 15 epochs, patience 5. Half of training views use random 50-100% spatial
crops; retained center coordinates are translated, with half-open crop membership
to avoid double assignment at seams. Remaining views retain whole-photo context.
The hypothesis is better recognition of small ends in dense stacks. Validation
is never cropped: use the original 1,332 full photos at float32, batch one, and
select for exact counts. Preserve the frozen 1,330-image test set. Use the same
validation reference rather than dropping hard/nested photos to raise the score.

After this run, inspect density errors against the baseline. A validation-only
tile comparison is justified after crop adaptation if full-photo dense counts
remain weak. No tile mode is promoted without evidence. Twenty-two tests pass,
including crop seam membership and protection against cropping validation.
No production-readiness claim follows from the clarified scope alone.

## Crop rejection and box-detector comparison, 2026-09-16T12:36:30.312285+00:00

The crop experiment stopped normally after six epochs (best epoch one): 25.38%
exact counts, MAE 8.90. Full validation diagnostics reproduced its recorded score.
Context-tile inference after crop adaptation reached only 14.26% exact, MAE 10.66.
Reject both for the primary exact-count objective. Preserve v4_float32 at 28.15%
exact / MAE 9.33. Do not repeat crop/resolution variants without new evidence.

The next distinct bounded comparison uses box supervision already present in
the labels, rather than only Gaussian center targets. Torchvision Faster R-CNN
ResNet50 FPN V2 starts from the official COCO weights. It learns pipe/background
classification and box regression with features at multiple spatial scales.
Configuration: 640px maximum image side, batch 2, AdamW LR 0.0002 for heads and
0.00002 for backbone, up to eight epochs, patience three, float32 batch-one
validation, exact-count threshold/checkpoint selection. Early pretrained ResNet
layers are frozen; BatchNorm running statistics are fixed for small batches.

Dense-scene settings explicitly replace the default 100 detections: 2,000 final
boxes, 4,000 post-NMS proposals at inference, anchors 16/32/64/128/256 pixels,
box NMS IoU 0.5. Training labels have at most 1,523 boxes; validation at most 1,121.
At 640px, minimum box-side percentiles are 7.85/15/23/32.62/60.8 at 5/25/50/75/95%.
All train/validation boxes passed conversion checks; no labels were removed.
No reserved-test images or labels were loaded for this experiment.

Twenty-five tests passed. CUDA smoke training succeeded at 128px and at 640px
on two images with 1,523 labels each. The latter checkpoint reloaded offline and
reproduced both validation fixture counts. Smoke scores are not accuracy evidence.
Official weights SHA256: dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf.
Reference: https://docs.pytorch.org/vision/0.26/_modules/torchvision/models/detection/faster_rcnn.html
New detector checkpoints use scripts/predict_detector.py, not the center-heatmap
predict.py. The saved v4 development package remains unchanged. Compare full frozen
validation before promoting the detector; do not evaluate reserved test merely
because training finishes. No production-readiness claim is warranted yet.

## Detector failure diagnosed and corrected, 2026-09-16T13:14:47.598720+00:00

pipe_detector_v6 failed during epoch one after at least 100 batches: decoded image
dimensions disagreed with COCO metadata. No epoch/checkpoint was completed. This
was a data-geometry failure, not a GPU failure. The v4 baseline remains preserved.

Decoded all 11,982 training/validation images, excluding reserved test: 20 size
mismatches (17 training, 3 validation). Every affected metadata size was reversed
relative to encoded pixels; ignoring EXIF orientation did not resolve it. Twelve
had nonempty labels. Compared direct coordinates, clockwise and counterclockwise
rotation, and dimension scaling overlays for every labeled mismatch. A 90-degree
counterclockwise label-coordinate transform aligned the boxes in all twelve.
The other eight records retain their empty label sets and use decoded dimensions.
One reviewed scene visibly has partial annotations; spatial alignment does not
prove completeness. This remains a label-quality limitation.

data/geometry_overrides_v1.json is an explicit, SHA256-keyed adapter containing
only the audited images, dimensions, counts and transform. The loader verifies
image bytes and expected geometry; unknown mismatches still fail with a specific
filename. Raw photos, frozen manifests, split groups, and counts were not changed.
The two nonempty affected validation images keep their existing counts, so count
accuracy remains comparable; box-location accuracy would need corrected geometry.
The adapter is included and hashed in detector checkpoints and source snapshots.

Twenty-six tests pass, including transformed box positions, count preservation,
and rejection of unapproved geometry. A full decoded-data preflight and a CUDA
smoke run on corrected and empty records precede restarting the same bounded
detector comparison in pipe_detector_v6_geometry (eight epochs, patience three).
The failed attempt is preserved. No reserved-test data was evaluated or corrected.

Geometry preflight completed: 10,650 training images / 1,529,564 labels and 1,332 validation images / 206,171 labels decoded successfully with counts preserved. Corrected-coordinate CUDA smoke training/validation passed. The repaired detector comparison has restarted as pipe_detector_v6_geometry.


## Completed detector and suppression calibration, 2026-09-16T14:57:26.953338+00:00

pipe_detector_v6_geometry completed eight epochs normally; epoch eight is best.
Original NMS IoU 0.5: 29.05% exact, 45.05% within one, MAE 16.30. This is only
12 more exactly correct images than the 28.15% heatmap baseline, with worse MAE.
Do not call it an unqualified improvement. Detector MAE improves over the heatmap
model in the 11-50 and 51-100 bands (2.93 vs 3.55; 4.31 vs 5.22), but worsens
above 100, especially 501+ (82.31 vs 32.09). Comparison and density/source reports:
runs/supervision/detector_comparison.json. Several major false-positive cases are
noisy collages; side-by-side domain scope does not resolve these quality issues.

A validation-only NMS IoU sweep (0.2 through 0.7) used one forward pass per image.
The original 0.5 setting reproduced its recorded score. Stronger suppression at
0.2 with threshold 0.455 reached 29.88% exact (398/1332), 46.92% within one,
MAE 10.40. This reduces duplicate-box errors, but is still far from production
accuracy and still has worse MAE than the heatmap model's 9.33. Both models remain
preserved. Calibrated detector: runs/pipe_detector_v6_calibrated/candidate.pt.
Sixteen normal/dense/error cases reproduced their diagnostic counts through
checkpoint reload and native inference with stored NMS settings. Tests: 27 pass;
checkpoint-initialized CUDA smoke training also passed. The initialization guard
rejects unknown/outside-development identities or changed geometry metadata.

One lower-learning-rate continuation is justified by the last-epoch validation
trend: detector MAE declined 84.20 -> 77.56 -> 19.33 -> 16.30 over epochs 5-8
while final exact counts improved. Next: pipe_detector_v7_finetune, initialized
from the calibrated detector, 640px, batch 2, LR 0.00005 (backbone 0.000005),
at most eight additional epochs, patience three, NMS IoU 0.2, float32 batch-one
validation. Optimizer/scheduler/scaler reset; this is a new fine-tune, not an exact
resume. Parent checkpoint identity is recorded. No blind further extensions.

Asked an optional question about typical and maximum pipes per actual photo to
prioritize relevant density checks. No reply is required to continue this existing
bounded comparison; retain the full frozen validation reference in the meantime.
No new operating count range is assumed. Test remains unused; no production approval.

## Completed v7 and side-by-side error review, 2026-09-16T18:40:04.810636+00:00

The bounded fine-tune stopped normally after five epochs; epoch two was best.
Exact counts 397/1332 (29.8048%), within one 45.9459%, MAE 8.35285, RMSE
29.71888 at threshold 0.43 and NMS IoU 0.2. This lowers MAE about 20% from
v6 calibrated, with one fewer exact image. Both candidates are retained.
No further full training is active. The reserved test remains unused.

Ten selected errors from distinct validation groups were visually inspected,
not used as a representative accuracy sample. Partial ends are inconsistently
labeled; some visible ends are omitted. Real model problems include duplicate
boxes, small distant-end misses and a false pipe-body detection. One review
image contains nesting and cannot represent a pure side-by-side benchmark.
No relabeling, split changes or post-hoc removal of hard examples was performed.
Review: C:\Users\Owner\Documents\Codex\2026-09-15\okay-look-though-my-folder-let\outputs\Side-by-side-error-review\index.html

Next: audit exact duplicate training boxes and partial-end conventions before
versioning verified annotation corrections and running another bounded experiment.
Do not assume labels explain all errors, or claim production/domain accuracy
from selected examples. Side-by-side scope is recorded; nesting is out of scope.

## Training-label source audit and v7 packaging, 2026-09-16T19:19:22.823145+00:00

No active Python training process was found; v7 logs/history/checkpoints confirm
normal five-epoch completion. No additional training was launched.

Verified all 49 training records with 409 repeated boxes against original COCO
entries and image SHA256 hashes. Every repeated box is identical in the raw
source, not a clipping/preparation artifact. Top examples have 203 annotations
but 51 distinct boxes, with most ends repeated four times. A separate versioned
training manifest removes exact repeats while preserving images and group IDs:
C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc\data\training_revisions\exact_duplicate_boxes_v1\train.jsonl
Round-trip verification passed: 10,650 images, 1,529,155 labels. This staged
revision has not been used to train. Geometry-override overlap: 0.
Original raw data and frozen manifests remain untouched. This affects only
0.027% of training labels, so do not present it as the explanation of all errors
or automatically launch another run solely because it is available.

Packaged v7 with optimizer removed, identical weights/configuration, bundled
inference source, provenance, metrics and 16 validation regression fixtures:
C:\Users\Owner\Documents\Codex\2026-09-15\okay-look-though-my-folder-let\outputs\Pipe-detector-v7-development
All 16 CUDA CLI counts matched prior validation predictions; repeated GPU
inference also matched. CPU matched 15/16: fixture 11 gives CPU 51 versus CUDA
50. Diagnostic: cuDNN TF32 enabled yields score 0.42993456 just below threshold
0.43; disabling TF32 yields 0.43020958 and 51, matching CPU. The package runner
now explicitly pins cuDNN TF32 on and CUDA matrix-multiply TF32 off to match the
audited reference. A test starting with opposite cuDNN setting reproduced 50.
Float32 tensor dtype alone does not imply strict full-precision GPU arithmetic.
CPU count parity is NOT approved; other GPU/runtime combinations remain untested.
No weights or thresholds were changed to hide this difference. Local warmed
GPU forward median was 37.94 ms over 16 cases, excluding decode,
tensor creation, startup and output writing; not an Azure/end-to-end SLA.
Package checksum verification passed. This remains a DEVELOPMENT package,
not a production release. Test data remains unused.

## Training annotation-conflict review, 2026-09-16T19:53:51.896729+00:00

Reconstructed every training conflict from source COCO entries: 113 retained
training images, 80 with differing counts, 313 source image copies hash-verified.
Every selected training label list reproduced exactly. Eight largest differences
from distinct groups were visually reviewed without model predictions.

The 7-versus-68 and 360-versus-531 examples expose different target definitions:
round ends only versus round plus square/rectangular tubes. Other examples differ
on partially cropped ends and separate background stacks. Some sets also omit
clearly visible ends. Max-annotation selection is not equivalent to verified truth.
Do not silently switch counts or declare either export correct. Nesting remains
out of scope, as the user already clarified. Illustrated evidence:
C:\Users\Owner\Documents\Codex\2026-09-15\okay-look-though-my-folder-let\outputs\Training-conflict-audit\index.html

Ask the user to define shape and boundary treatment once, with concrete examples.
No ambiguous annotations changed; original frozen train/val/test maintained.
While waiting, equal-count variants can still be audited for spatial errors
without guessing the user's counting target. No additional training launched.

Follow-up spatial audit, 2026-09-16T19:55:46.419493+00:00: ranked all 33 equal-count training conflicts
by symmetric best-box IoU and visually inspected the four least-agreeing distinct
groups. All four show the same single pipe end with varying box extent/tightness;
no new gross coordinate transform fault was established. No labels changed.
Two concise user questions are now pending about shapes and partial ends.
Await these answers before revising ambiguous counting targets. Do not repeat
the questions on every heartbeat or launch speculative training while waiting.
No active training job; frozen validation and reserved test remain unchanged.

## User-requested continuation and resolution diagnostic, 2026-09-16T21:45:42.326216+00:00

The user said "keep going". Resume independent improvement work without
interpreting this as an answer to the pending shape/boundary questions. Existing
labels and counting conventions stay unchanged during the next comparison.

One full validation-only check resized the existing v7 detector to 960px, with
weights/NMS/math settings unchanged. At the original threshold, exact 25.90%,
MAE 6.3926. At a validation-calibrated 0.47 threshold: exact 27.4024%, within
one 45.1952%, MAE 6.3296, RMSE 20.0930. The 640px baseline remains exact
29.8048%, MAE 8.3529. Higher resolution reduces large errors but is not a
unqualified improvement; do not replace the preserved candidate.

For 501+ labels per photo, MAE improved 47.2857 -> 26.3377 (about 44%).
For 251-500 it improved 12.0544 -> 8.1213. Small-image bands regress.
This meets the predeclared rule for one bounded adaptation: at least 20% dense
MAE reduction without overall MAE regression. Report is in
runs/pipe_detector_v7_960_diagnostic/report.json; all 1,332 validation images
were used, baseline cached scores reproduced checkpoint metrics, no test use.

A dense-batch smoke check at 960px with two 1,523-label training images passed
forward/backward/optimizer step; peak allocation 8.55 GiB on RTX 5090.
Plan: pipe_detector_v8_960, initialize v7 best, batch two, head LR 2e-5 and
backbone LR 2e-6, max six epochs/patience two, NMS 0.2; new optimizer state.
Keep original frozen training/validation labels to isolate this experiment;
the staged duplicate-label revision is not applied. Compare exact accuracy,
MAE and density errors before selecting anything. No blind extensions.

The bounded v8_960 supervisor launched successfully. Previous completed v7 lock/status were archived under runs/supervision/v7_completed_before_v8_960. Six-epoch maximum, patience two; no other GPU training run was active at launch. Existing heartbeat supervision continues.

## 95% target and architecture exploration, 2026-09-16T22:08:48.372444+00:00

User explicitly authorized other architectures and set a 95%+ goal. The working
metric stated to the user is at least 95% exactly counted photos. This replaces
the older provisional 99% target; ACCEPTANCE_CRITERIA.json is authoritative.
The already-running v8 supervisor loaded older metadata at startup and may write
a stale 99% field when it exits; reconcile that with the criteria file rather
than interpreting it as a user change. Future supervisor writes load the new file.

V8 continues; no second GPU job launched. Prepared FCOSResNet50FPN as the next
architecture, using cached official COCO initialization with verified SHA256
99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7.
FCOS retains pretrained feature/box heads, replaces only the final class logits,
uses frozen batch-norm consistently for pretrained and offline construction,
permits 2,000 detections/10,000 candidates per level, and filters class 0 from
counts. Generic detector train/reload/inference and supervisor now support it.
Future training records explicit cuDNN/matmul TF32 settings; inference honors
these recorded fields. Existing v8 process and packaged candidates are unchanged.

28 CPU tests pass, including exact offline FCOS reload outputs and rejection of
cross-architecture initialization. Official-weight load plus a positive/empty
CPU training batch and optimizer step passed. GPU dense-batch preflight is
deferred until v8 releases the GPU. Queue and bounded launch settings are in
runs/supervision/architecture_queue.json: 960px, batch two, LR 1e-4, max 12
epochs, patience four, NMS .3, original frozen train/validation labels.

RF-DETR is a later transformer candidate, but the documented 300-query defaults
would cap dense counts. Raising query/output capacity requires adaptation and
more memory; tiling needs boundary/duplicate handling. YOLO26 was also researched
as an alternative. No benchmark mAP is substituted for exact-count accuracy,
no package changes were made to the active environment, and no production result
is implied by architecture choice. Test remains reserved.


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
