import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import torch

project = Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
output = Path(__file__).resolve().parents[1]/'outputs'
destination = output/'Pipe-counter-development-candidate'
destination.mkdir(exist_ok=False)
report_path = project/'runs/validation_diagnostics/v4_best_float32_single/512/report.json'
report = json.loads(report_path.read_text())
original_path = project/'runs/pipe_center_v4_resnet18_exact/best.pt'
checkpoint = torch.load(original_path, map_location='cpu', weights_only=False)
checkpoint.pop('optimizer', None)
checkpoint['original_training_metrics'] = checkpoint['metrics']
checkpoint['original_training_threshold'] = checkpoint['threshold']
checkpoint['metrics'] = report['optimized_for_exact_count']
checkpoint['threshold'] = checkpoint['metrics']['threshold']
checkpoint['config']['inference_precision'] = 'float32'
checkpoint['config']['inference_batch_size'] = 1
checkpoint['calibration'] = {'validation_report':str(report_path),
    'parent_checkpoint_sha256':hashlib.sha256(original_path.read_bytes()).hexdigest(),
    'validation_manifest_sha256':hashlib.sha256((project/'data/processed_v2/val.jsonl').read_bytes()).hexdigest(),
    'calibrated_at':datetime.now(timezone.utc).isoformat(),
    'test_evaluated':False, 'production_ready':False,
    'note':'Weights unchanged; precision and threshold calibrated using validation only. Development candidate, not a production release.'}
torch.save(checkpoint, destination/'candidate.pt')
shutil.copytree(project/'src/pipe_counter', destination/'src/pipe_counter', ignore=shutil.ignore_patterns('__pycache__'))
(destination/'scripts').mkdir()
for name in ('predict.py','evaluate.py'):
    shutil.copy2(project/'scripts'/name, destination/'scripts'/name)
for name in ('pyproject.toml', 'requirements-local-lock.txt'):
    shutil.copy2(project/name,destination/name)
shutil.copy2(report_path,destination/'validation_report.json')
shutil.copy2(project/'runs/pipe_center_v4_resnet18_exact/provenance.json',destination/'training_provenance.json')
rows = list(csv.DictReader((report_path.parent/'per_image.csv').open(encoding='utf-8')))
examples = rows[:12]
(destination/'regression_examples').mkdir()
for i,row in enumerate(examples):
    source = Path(row['image'])
    target = destination/'regression_examples'/f'{i:02d}{source.suffix}'
    shutil.copy2(source,target)
    row['packaged_image'] = str(target.relative_to(destination))
(destination/'regression_examples.json').write_text(json.dumps(examples,indent=2))
(destination/'README.md').write_text('''# Pipe counter development candidate

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
''',encoding='utf-8')
print(destination)
