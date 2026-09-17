import json
from datetime import datetime, timezone
from pathlib import Path

project = Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
outputs = Path(__file__).resolve().parents[1] / 'outputs'
supervision = project / 'runs/supervision'
state = json.loads((supervision / 'status.json').read_text())
assert state['phase'] == 'training' and state['run_dir'].endswith('pipe_center_v4_resnet18_exact')
prior = json.loads((supervision / 'v3_completed_status.json').read_text())
experiments = json.loads((supervision / 'experiments.json').read_text())
experiments['updated_at'] = datetime.now(timezone.utc).isoformat()
if not any(r['experiment'] == 'pipe_center_v3_768_exact' for r in experiments['results']):
    experiments['results'].append({'experiment': 'pipe_center_v3_768_exact', 'resolution': '768',
        'metrics': prior['best_validation_metrics'], 'decision': 'Rejected: lower exact-count accuracy than the preserved 512 baseline.'})
experiments['next_experiment'] = {
    'run': 'pipe_center_v4_resnet18_exact', 'status': 'training',
    'hypothesis': 'An ImageNet-pretrained ResNet18 encoder may improve feature learning versus the small from-scratch encoder.',
    'architecture': 'ResNet18CenterNet', 'image_size': 512, 'batch_size': 16,
    'epochs_max': 30, 'patience': 6, 'learning_rate': 0.0003, 'encoder_lr_factor': 0.1,
    'selection_metric': 'exact_count_accuracy', 'baseline_exact_accuracy': 0.20195195195195195,
    'baseline_mae': 10.903153153153154,
    'stop_policy': 'If this does not materially improve validation accuracy, prioritize label adjudication and target representation rather than another speculative training run.'}
(supervision / 'experiments.json').write_text(json.dumps(experiments, indent=2), encoding='utf-8')
update = '''
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
'''
with (project / 'PRODUCTION_READINESS.md').open('a', encoding='utf-8') as handle:
    handle.write(update)
(outputs / 'Production-readiness-plan.md').write_text((project / 'PRODUCTION_READINESS.md').read_text(), encoding='utf-8')
(outputs / 'Validation-progress.md').write_text('''# Pipe counter: supervised accuracy review

Status: not production-ready. Reserved test data remains unevaluated.

| Validation experiment | Exactly correct image counts | Average absolute count error |
|---|---:|---:|
| Preserved 512-pixel baseline, tuned threshold | 20.20% | 10.90 |
| 768-pixel fine-tune, best epoch 10 | 16.97% | 13.26 |
| 768 input without adaptation | 11.56% | 28.27 |
| Context tiles | 10.21% | 12.21 |

These results use 1,332 validation images and validation-tuned thresholds.
They are not held-out test or production accuracy. The higher-resolution
fine-tune was rejected and the better baseline is preserved.

A pretrained ResNet18 experiment started on the RTX 5090 at 04:02 UTC on
September 16. It uses the same split and labels, 512-pixel input, and at most
30 epochs with early stopping. Eighteen tests and a CUDA training smoke check
passed before launch. Its results will determine whether it replaces the baseline.

Data quality is a substantial unresolved issue: 100 identical-image groups
have differing count labels. Nested-pipe counting rules also need clarification.
No labels were silently rewritten. If this experiment fails to materially
improve results, the next work will focus on label adjudication and how targets
are represented rather than another speculative training run.

Monitoring remains active every 30 minutes, with notifications for meaningful
results, failures, or decisions. Fully automatic production readiness is not
established by the current evidence.
''', encoding='utf-8')
print('Recorded v3 rejection, v4 training, and updated user-facing reports.')
