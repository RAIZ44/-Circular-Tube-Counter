import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
outputs=Path(__file__).resolve().parents[1]/'outputs'
supervision=project/'runs/supervision'
state=json.loads((supervision/'status.json').read_text())
assert state['phase']=='validation_review' and state['training_exit_code']==0
assert state['run_dir'].endswith('pipe_detector_v6_geometry')
comparison=json.loads((supervision/'detector_comparison.json').read_text())
nms=json.loads((project/'runs/validation_diagnostics/v6_nms/report.json').read_text())
metrics=nms['variants']['0.2']
experiments_path=supervision/'experiments.json'
experiments=json.loads(experiments_path.read_text(encoding='utf-8-sig'))
now=datetime.now(timezone.utc).isoformat()
experiments['updated_at']=now
experiments['results'].append({'experiment':'pipe_detector_v6_geometry','metrics':state['best_validation_metrics'],
    'decision':'Completed eight epochs. Slight exact-count improvement, but worse MAE; preserve as an alternative, not an unqualified promotion.'})
experiments['results'].append({'experiment':'pipe_detector_v6_calibrated','metrics':metrics,'nms_iou':.2,
    'decision':'Best exact validation result to date; MAE still worse than heatmap baseline. Both candidates retained.'})
experiments['next_experiment']={'run':'pipe_detector_v7_finetune','status':'prepared',
    'hypothesis':'The completed detector continued improving through epoch eight (MAE 84.2 at epoch five to 16.3 at epoch eight); one lower-rate continuation may improve localization and confidence for dense images.',
    'initial_checkpoint':'runs/pipe_detector_v6_calibrated/candidate.pt','optimizer_reset':True,
    'epochs_max':8,'patience':3,'image_size':640,'batch_size':2,'learning_rate':.00005,'backbone_lr':.000005,
    'nms_iou':.2,'validation_batch_size':1,'validation_precision':'float32',
    'baseline_exact_accuracy':metrics['exact_count_accuracy'],'baseline_mae':metrics['mean_absolute_error'],
    'heatmap_baseline_exact_accuracy':.28153153153153154,'heatmap_baseline_mae':9.332582582582583,
    'stop_policy':'One bounded continuation justified by recent validation trend. Do not repeatedly extend training or claim readiness from a small exact-count gain; require error-distribution review, including MAE and dense scenes.'}
experiments['next_action']='Compare the bounded fine-tune against both preserved candidates. Keep reserved test unused. If no meaningful improvement, focus on audited labels and matched operating-domain validation instead of another epoch extension.'
experiments_path.write_text(json.dumps(experiments,indent=2))
update=f'''
## Completed detector and suppression calibration, {now}

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
'''
with (project/'PRODUCTION_READINESS.md').open('a',encoding='utf-8') as handle:handle.write(update)
shutil.copy2(project/'PRODUCTION_READINESS.md',outputs/'Production-readiness-plan.md')
(outputs/'Validation-progress.md').write_text('''# Pipe counter progress

**Development continues; no model is approved for fully automatic production.**

All figures below are tuned on the same 1,332 validation photos. The reserved
test set remains unused.

| Preserved candidate | Exactly correct photos | Within one pipe | Average absolute error |
|---|---:|---:|---:|
| Center model, full precision | 28.15% | 41.82% | 9.33 pipes |
| Box detector, original settings | 29.05% | 45.05% | 16.30 pipes |
| Box detector, improved duplicate suppression | 29.88% | 46.92% | 10.40 pipes |

The detector is slightly better at exact counts, while the center model still
has the lower average error. Both are preserved. The detector performs better
on many 11-100-pipe scenes; dense scenes remain particularly difficult.

Twenty-six earlier geometry checks plus the initialization guard give 27 passing
tests. All 11,982 training/validation images passed a decoded-data preflight.
The calibrated detector reproduces sixteen diagnostic counts after reloading.
These checks establish implementation consistency, not production accuracy.

One bounded fine-tune at a lower learning rate is being started because the
detector was still improving at the end of its first eight epochs. It is limited
to eight further epochs with early stopping. Monitoring remains active.

The user clarified side-by-side pipes; nesting is outside the intended case.
The existing heatmap development package remains a preserved comparison model.
No Azure deployment or reserved-test evaluation has been performed.
''',encoding='utf-8')
print('Recorded completed detector, calibrated alternative, and bounded follow-up.')
