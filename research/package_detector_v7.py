import csv, hashlib, json, shutil
from pathlib import Path
from datetime import datetime, timezone
import torch
from pipe_counter.utils import read_jsonl

project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
workspace=Path(__file__).resolve().parents[1]
run=project/'runs/pipe_detector_v7_finetune'
out=workspace/'outputs/Pipe-detector-v7-development'
out.mkdir(exist_ok=False)
original=run/'best.pt'
checkpoint=torch.load(original,map_location='cpu',weights_only=False)
checkpoint.pop('optimizer',None)
checkpoint['package_provenance']={
    'created_at':datetime.now(timezone.utc).isoformat(),
    'parent_checkpoint_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
    'weights_and_inference_settings_changed':False,
    'test_evaluated':False,'production_ready':False,
    'note':'Optimizer removed for inference packaging. No new training, calibration or label changes.'}
torch.save(checkpoint,out/'candidate.pt')
shutil.copytree(project/'src/pipe_counter',out/'src/pipe_counter',ignore=shutil.ignore_patterns('__pycache__'))
(out/'scripts').mkdir()
original_script=(project/'scripts/predict_detector.py').read_text(encoding='utf-8')
bootstrap='import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))\n'
(out/'scripts/predict_detector.py').write_text(bootstrap+original_script,encoding='utf-8')
for name in ['requirements-local-lock.txt']:
    shutil.copy2(project/name,out/name)
for source,target in [(run/'provenance.json','training_provenance.json'),
                      (run/'geometry_overrides.json','geometry_overrides.json'),
                      (run/'best_validation_metrics.json','validation_metrics.json'),
                      (project/'runs/supervision/detector_v7_comparison.json','validation_comparison.json'),
                      (project/'OPERATING_SCOPE.json','operating_scope.json')]:
    shutil.copy2(source,out/target)
records=read_jsonl(project/'data/processed_v2/val.jsonl')
rows=list(csv.DictReader((run/'best_validation_per_image.csv').open(encoding='utf-8')))
assert len(rows)==len(records)==1332
for row,record in zip(rows,records):
    assert Path(row['image'])==Path(record['image_path'])
    assert int(row['true_count'])==len(record['boxes'])
selected=[]
groups=set()
def select(i):
    if records[i]['split_group'] not in groups:
        selected.append(i);groups.add(records[i]['split_group'])
select(0)
for lower,upper in [(0,0),(1,10),(11,50),(51,100),(101,250),(251,500),(501,10000)]:
    select(next(i for i,r in enumerate(records) if lower<=len(r['boxes'])<=upper))
geometry=json.loads((run/'geometry_overrides.json').read_text())['images']
for i,r in enumerate(records):
    if r['sha256'] in geometry:select(i)
select(max(range(len(records)),key=lambda i:len(records[i]['boxes'])))
for i in range(len(records)):
    if len(selected)>=16:break
    select(i)
(out/'regression_examples').mkdir()
examples=[]
for j,i in enumerate(selected):
    r=records[i]
    filename=f'{j:02}{Path(r["image_path"]).suffix}'
    shutil.copy2(r['image_path'],out/'regression_examples'/filename)
    examples.append({'file':f'regression_examples/{filename}','sha256':r['sha256'],
        'validation_index':i,'split_group':r['split_group'],
        'expected_prediction':int(rows[i]['predicted_count']),
        'existing_label_count':len(r['boxes']),
        'note':'Development regression fixture, not an independent accuracy case.'})
(out/'regression_examples.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
(out/'README.md').write_text('''# Pipe detector v7 — development package

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
The output contains a CSV count summary and a JSON file for every photo with
boxes and scores. The reported detection-limit flag means the internal 2,000-box
limit was reached; the model cannot establish that a capped count is complete.

## Fixed inference behavior

Faster R-CNN ResNet50 FPN V2, float32, one photo at a time, RGB values in [0,1].
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

Known limitations include inconsistent labels at cropped image boundaries,
missing or duplicate labels, duplicate detections, small-end misses and dense
scenes. Existing grouped splits reduce known related-image overlap but cannot
rule out shared collage components or renamed scenes. No accuracy claim is made
for an automatically filtered side-by-side subset. No Azure deployment has
occurred; other runtimes or accelerators require parity and performance checks.

candidate.pt retains development provenance and geometry metadata; optimizer
state was removed without changing weights or inference settings. Original
training checkpoints remain preserved. SHA256SUMS.json records package hashes.
''',encoding='utf-8')
print(json.dumps({'package':str(out),'fixtures':len(examples),'bytes':(out/'candidate.pt').stat().st_size}))
