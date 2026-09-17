# Field benchmark

Optional future workflow. The user has confirmed the four existing exports are
the complete available image set. The approved current benchmark uses
data/processed_v2: 10,650 training, 1,332 validation, and 1,330 test images.
Additional field photos are not a prerequisite for training and evaluating that
split. The steps below apply if separately captured photos become available.
No new-field accuracy has been measured.

Capture separate sessions covering the actual camera position, lighting, pipe
sizes, sparse and dense stacks, partial occlusion, and empty scenes. Keep entire
capture sessions out of training and model selection. Several views of the same
stack are correlated; record them under the same scene_id.

Agree on the counting rule before labeling: whether partial pipe ends at the
image border count, and which stack or region is in scope. For this model, use
visible pipe ends across the image. Physically hidden pipes cannot be verified
from a photograph alone. Record ambiguous images for review, not guessed counts.

From the project directory, use the parent environment:

```powershell
& ..\.venv\Scripts\python.exe scripts\field_benchmark.py init --images "C:\path\to\field_photos" --output data\field_benchmark\review.csv
```

Enter true_count, reviewed=yes, and scene_id for every row in review.csv.
The image hash binds each review row to the original image. Creating a review
sheet again at the same path is refused to protect manual work.

Generate predictions using the checkpoint's fixed validation threshold:

```powershell
& ..\.venv\Scripts\python.exe scripts\predict.py --checkpoint runs\pipe_center_v1\best.pt --input "C:\path\to\field_photos" --output-dir runs\field_benchmark_predictions
& ..\.venv\Scripts\python.exe scripts\field_benchmark.py score --review data\field_benchmark\review.csv --predictions runs\field_benchmark_predictions\predictions.csv --development-manifests data\processed\train.jsonl data\processed\val.jsonl --output runs\field_benchmark\metrics.json
```

For a newly trained v2 checkpoint, use the corresponding processed_v2 training
and validation manifests instead. Always include every dataset used to train,
tune, or select the model, and keep the evaluation threshold fixed.

Scoring refuses blank/unreviewed counts, changed images, duplicate images,
missing or extra predictions, and detected development-data overlap. It reports
exact-count accuracy, within-one accuracy, average absolute error, signed error,
and per-image errors. Automated hashes and filename checks do not detect every
crop, collage, alternate filename, or related camera view; verify scene
independence yourself. The script cannot establish that supplied manifests
describe all development data for an externally supplied checkpoint.
