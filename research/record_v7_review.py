import json, html, shutil
from pathlib import Path
from datetime import datetime, timezone

workspace = Path(__file__).resolve().parents[1]
project = Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
now = datetime.now(timezone.utc).isoformat()
audit = json.loads((workspace / 'work/v7_error_audit/index.json').read_text(encoding='utf-8-sig'))
notes = [
    'Extra detection on a narrowly cropped left-edge end. Border visibility needs a consistent annotation rule; this is not nesting.',
    'Several partial ends along the image boundary are detected but absent from labels. Some very narrow boundary pieces remain ambiguous.',
    'The four main ends agree; two additional cropped ends at top and bottom are visible and detected but unlabeled.',
    'An additional partly visible end at the far right is detected but unlabeled.',
    'Capped ends at an oblique angle are missed or merged. The image also contains artificial-looking black seams, so model and image-quality issues coexist.',
    'An extra short pipe-like object in the foreground is detected but unlabeled. Whether loose fittings belong in the count needs an explicit dataset convention.',
    'Multiple orange boxes overlap the same left/bottom pipe ends, with an additional small box on a rim. Duplicate detections are a real model failure.',
    'Small distant ends near the upper edge are missed. One small pipe is also nested at lower left, making this a mixed-scope example rather than a clean side-by-side benchmark.',
    'Two lower-corner cropped ends are detected but unlabeled, although other boundary ends are labeled.',
    'Several visible ends lack labels, and the model also produces an incorrect long box along a pipe body. Both annotation and model failures appear.'
]
out = workspace / 'outputs/Side-by-side-error-review'
out.mkdir(parents=True, exist_ok=True)
cards = []
for case, note in zip(audit['cases'], notes):
    filename = f"{case['case']:02}.jpg"
    shutil.copy2(case['panel'], out / filename)
    case['visual_observation'] = note
    case['observation_status'] = 'AI visual review; no ground-truth changes made'
    cards.append(f'<section><h2>Case {case["case"]:02}: labels {case["labels"]}, model {case["predicted"]}</h2><p>{html.escape(note)}</p><a href="{filename}"><img src="{filename}" alt="Original, labels and detections for case {case["case"]}"></a></section>')
audit['reviewed_at'] = now
audit['limitations'] = 'Selected errors, not a representative sample or domain accuracy estimate. Visual observations are provisional; labels, splits and thresholds are unchanged.'
(out / 'review.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
(out / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Side-by-side pipe error review</title><style>body{font:17px system-ui;margin:32px auto;max-width:1500px;padding:0 20px;color:#18212d;background:#f5f7fa}section{background:white;padding:20px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}h1{font-size:30px}p{max-width:1000px;line-height:1.5}</style><h1>Side-by-side pipe error review</h1><p>Ten selected validation errors from distinct groups, mostly small side-by-side scenes. This is a diagnostic review, not an accuracy estimate. Green boxes are existing labels; orange boxes are model detections. Click an image to inspect it at full size.</p><p>Nesting is outside the intended use. These examples show that inconsistent treatment of partial ends, missing labels, duplicate detections and small-end misses remain relevant. No labels have been changed from this visual review.</p>' + ''.join(cards) + '</html>', encoding='utf-8')

progress = '''# Pipe counter progress

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
'''
(workspace / 'outputs/Validation-progress.md').write_text(progress, encoding='utf-8')

status_path = project / 'runs/supervision/status.json'
status = json.loads(status_path.read_text(encoding='utf-8-sig'))
assert Path(status['run_dir']).name == 'pipe_detector_v7_finetune'
assert status.get('training_exit_code') == 0
status.update(phase='data_quality_review', updated_at=now, active_training=False,
    next_action='Audit exact duplicate training annotations and partial-end conventions; version verified corrections before a bounded experiment. No blind training extension. Keep reserved test unused.',
    error_review=str(out / 'index.html'))
status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
exp_path = project / 'runs/supervision/experiments.json'
experiments = json.loads(exp_path.read_text(encoding='utf-8-sig'))
entry = dict(experiments.get('next_experiment') or {})
entry.update(experiment='pipe_detector_v7_finetune', status='completed', epochs_completed=5, best_epoch=2,
    metrics=status['best_validation_metrics'], decision='Preserve as lower-MAE alternative; one fewer exact validation photo than v6 calibrated. Stop blind epoch extensions and review data/model errors.')
experiments['results'] = [x for x in experiments['results'] if x.get('experiment') != entry['experiment']] + [entry]
experiments.update(updated_at=now, next_experiment=None, next_action=status['next_action'])
exp_path.write_text(json.dumps(experiments, indent=2), encoding='utf-8')
readiness = project / 'PRODUCTION_READINESS.md'
addition = f'''\n## Completed v7 and side-by-side error review, {now}\n\nThe bounded fine-tune stopped normally after five epochs; epoch two was best.\nExact counts 397/1332 (29.8048%), within one 45.9459%, MAE 8.35285, RMSE\n29.71888 at threshold 0.43 and NMS IoU 0.2. This lowers MAE about 20% from\nv6 calibrated, with one fewer exact image. Both candidates are retained.\nNo further full training is active. The reserved test remains unused.\n\nTen selected errors from distinct validation groups were visually inspected,\nnot used as a representative accuracy sample. Partial ends are inconsistently\nlabeled; some visible ends are omitted. Real model problems include duplicate\nboxes, small distant-end misses and a false pipe-body detection. One review\nimage contains nesting and cannot represent a pure side-by-side benchmark.\nNo relabeling, split changes or post-hoc removal of hard examples was performed.\nReview: {out / 'index.html'}\n\nNext: audit exact duplicate training boxes and partial-end conventions before\nversioning verified annotation corrections and running another bounded experiment.\nDo not assume labels explain all errors, or claim production/domain accuracy\nfrom selected examples. Side-by-side scope is recorded; nesting is out of scope.\n'''
with readiness.open('a', encoding='utf-8') as stream:
    stream.write(addition)
print(json.dumps({'status':status['phase'], 'report':str(out / 'index.html'), 'reviewed_cases':len(notes), 'test_evaluated':status['test_evaluated']}))
