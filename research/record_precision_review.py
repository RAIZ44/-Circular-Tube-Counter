import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

project = Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
outputs = Path(__file__).resolve().parents[1]/'outputs'
package = outputs/'Pipe-counter-development-candidate'
supervision = project/'runs/supervision'
now = datetime.now(timezone.utc).isoformat()
report = json.loads((package/'validation_report.json').read_text())
metrics = report['optimized_for_exact_count']
rows = list(csv.DictReader((project/'runs/validation_diagnostics/v4_best_float32_single/512/per_image.csv').open()))
errors = np.array([int(r['error']) for r in rows])
unflagged = [r for r in rows if r['annotation_conflict'] == 'False']
unflagged_exact = sum(int(r['error'])==0 for r in unflagged)/len(unflagged)
old = json.loads((supervision/'status.json').read_text())
assert old['phase']=='validation_review' and old['training_exit_code']==0
assert old['run_dir'].endswith('pipe_center_v4_resnet18_exact')
shutil.copy2(supervision/'status.json',supervision/'v4_completed_status.json')
preserved = project/'runs/pipe_center_v4_float32'
preserved.mkdir(exist_ok=False)
for name in ('candidate.pt','validation_report.json','reload_verification.json','runtime_verification.json','README.md'):
    shutil.copy2(package/name,preserved/name)
with zipfile.ZipFile(preserved/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for folder in ('src/pipe_counter','scripts','tests'):
        for path in (project/folder).glob('*.py'):
            archive.write(path,path.relative_to(project))
    archive.write(project/'pyproject.toml','pyproject.toml')
    archive.write(project/'requirements-local-lock.txt','requirements-local-lock.txt')
provenance = json.loads((project/'runs/pipe_center_v4_resnet18_exact/provenance.json').read_text())
current_hashes = {s:hashlib.sha256((project/f'data/processed_v2/{s}.jsonl').read_bytes()).hexdigest() for s in ('train','val','test')}
assert current_hashes == provenance['manifest_sha256']
state = dict(old)
state.update(phase='awaiting_count_definition',updated_at=now,
    current_candidate=str(preserved/'candidate.pt'),candidate_metrics=metrics,
    candidate_inference_precision='float32',candidate_inference_batch_size=1,
    production_ready=False,test_evaluated=False,manifest_hashes_unchanged=True,
    awaiting_user_decision='For nested pipes, count every visible pipe end including smaller inner pipes, or only the outer pipes?',
    next_action='Await the already-requested counting rule before adjudicating conflicting labels and launching label-dependent training. Preserve all current artifacts. Do not repeat the unchanged blocker notification or start speculative training. Accuracy remains inadequate even outside identified label conflicts.',
    review_sheet=str(outputs/'Pipe-label-review.html'),
    blocker_last_reported_at=now)
temporary = supervision/'status.tmp'
temporary.write_text(json.dumps(state,indent=2)); temporary.replace(supervision/'status.json')
experiments = json.loads((supervision/'experiments.json').read_text())
experiments['updated_at'] = now
experiments['results'].append({'experiment':'pipe_center_v4_resnet18_exact','resolution':'512','metrics':old['best_validation_metrics'],'decision':'Completed at epoch 27; original best at epoch 21. Retained for precision calibration.'})
for name in ('v4_best_float32_single','v4_last_float32_single','v2_best_float32_single'):
    path=project/f'runs/validation_diagnostics/{name}/512/report.json'
    r=json.loads(path.read_text())
    experiments['results'].append({'experiment':name,'resolution':'512','inference_precision':'float32','inference_batch_size':1,'metrics':r['optimized_for_exact_count'],'report':str(path)})
experiments['next_experiment'] = None
experiments['next_action'] = state['next_action']
(supervision/'experiments.json').write_text(json.dumps(experiments,indent=2))
update = f'''
## Precision repair and candidate review, {now}

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
41.82% within one, 95th-percentile absolute error {np.percentile(np.abs(errors),95):.1f},
worst error {np.abs(errors).max()} pipes. Images without a recorded identical-image
label conflict are only {100*unflagged_exact:.2f}% exact ({len(unflagged)} images);
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
'''
with (project/'PRODUCTION_READINESS.md').open('a',encoding='utf-8') as handle:handle.write(update)
shutil.copy2(project/'PRODUCTION_READINESS.md',outputs/'Production-readiness-plan.md')
summary = f'''# Pipe counter progress

**Current status: development candidate; not production-ready.**

The best current model counts exactly correctly on **28.15% (375/1,332)** of
validation images. It is within one pipe on 41.82%, with average absolute error
9.33 pipes. Threshold and checkpoint were selected using these same validation
images. The reserved test set remains unused.

| Candidate under matching full-precision settings | Exact counts | Average absolute error |
|---|---:|---:|
| Earlier model | 23.95% | 11.36 |
| Pretrained model, selected epoch 21 | 28.15% | 9.33 |
| Pretrained model, final epoch 27 | 27.33% | 8.67 |

A numerical-precision defect created duplicate neighboring peaks. Inference
settings are now explicit and consistent, with a regression test. Twenty tests
pass, and a CUDA training integration check passes. The packaged candidate loads
from its bundled code and reproduces twelve saved examples on both CPU and GPU.

The remaining accuracy gap is substantial. More than 500 labeled pipes per image
are only 2.60% exact. Images without recorded duplicate-image label conflicts are
still only {100*unflagged_exact:.2f}% exact, so correcting known conflicts alone
is not proven to solve the problem.

Ten training/validation examples with conflicting annotations are ready in
Pipe-label-review.html. The unresolved rule is whether smaller pipes nested
inside larger pipes count too. No labels were silently corrected. Label-dependent
training is waiting for that decision; no full training job is currently active.
Monitoring remains active and will stay quiet while the decision is unchanged.

The Pipe-counter-development-candidate folder contains weights, inference code,
dependency lock, validation evidence, and regression fixtures. It is a development
snapshot, not an Azure deployment or a model approved for automatic use.
'''
(outputs/'Validation-progress.md').write_text(summary,encoding='utf-8')
with (package/'README.md').open('a',encoding='utf-8') as handle:
    handle.write('\n## Verification\n\nTwenty project tests and a CUDA training integration check passed. The bundled\nsource reproduced all twelve regression counts on CUDA, and CPU matched all\ntwelve. See reload_verification.json and runtime_verification.json. These checks\nverify software consistency, not production accuracy.\n')
hashes={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(package/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps({'candidate':str(preserved/'candidate.pt'),'phase':state['phase'],'exact_images':int((errors==0).sum()),'p95_error':float(np.percentile(np.abs(errors),95)),'unflagged_exact':unflagged_exact,'frozen_manifests_verified':True}))
