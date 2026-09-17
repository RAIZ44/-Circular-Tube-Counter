from pathlib import Path
from datetime import datetime, timezone
import json
project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
workspace=Path(__file__).resolve().parents[1]
now=datetime.now(timezone.utc).isoformat()
report=json.loads((project/'runs/pipe_detector_v7_960_diagnostic/report.json').read_text())
smoke=json.loads((workspace/'work/detector_960_smoke.json').read_text())
assert report['meets_predeclared_adaptation_rule'] and smoke['finite_loss_and_optimizer_step']
plan={'run':'pipe_detector_v8_960','status':'planned','image_size':960,'batch_size':2,
    'initial_checkpoint':'runs/pipe_detector_v7_finetune/best.pt','learning_rate':2e-5,
    'backbone_learning_rate':2e-6,'epochs_max':6,'patience':2,'nms_iou':.2,
    'optimizer_reset':True,'selection_metric':'exact_count_accuracy',
    'train_manifest':'data/processed_v2/train.jsonl','validation_manifest':'data/processed_v2/val.jsonl',
    'annotation_revision_applied':False,
    'hypothesis':'Adapt the detector to higher-resolution small pipe ends after an unadapted comparison lowered overall and dense-scene error but lost exact-count accuracy.',
    'rationale':'960px MAE 6.3296 versus 8.3529; 501+ MAE 26.3377 versus 47.2857. Exact accuracy fell 29.8048% to 27.4024%; preserve existing reference and assess adaptation.',
    'stop_policy':'One bounded resolution adaptation, six epochs maximum with two stale epochs. Compare exact counts AND MAE/density errors. No automatic production promotion or further epoch extensions.',
    'test_evaluated':False}
exp_path=project/'runs/supervision/experiments.json'
experiments=json.loads(exp_path.read_text())
experiments['results'].append({'experiment':'pipe_detector_v7_960_diagnostic','metrics':report['calibrated_960'],
    'fixed_threshold_metrics':report['fixed_threshold_960'],'decision':'No promotion: exact accuracy regressed. Lower overall/dense error meets predeclared rule for one bounded resolution adaptation.'})
experiments.update(updated_at=now,next_experiment=plan,next_action='Launch and supervise the bounded 960px adaptation after successful dense-batch CUDA smoke check.')
exp_path.write_text(json.dumps(experiments,indent=2),encoding='utf-8')
(project/'runs/supervision/v8_960_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
with (project/'PRODUCTION_READINESS.md').open('a',encoding='utf-8') as stream:
    stream.write(f'''\n## User-requested continuation and resolution diagnostic, {now}\n\nThe user said "keep going". Resume independent improvement work without\ninterpreting this as an answer to the pending shape/boundary questions. Existing\nlabels and counting conventions stay unchanged during the next comparison.\n\nOne full validation-only check resized the existing v7 detector to 960px, with\nweights/NMS/math settings unchanged. At the original threshold, exact 25.90%,\nMAE 6.3926. At a validation-calibrated 0.47 threshold: exact 27.4024%, within\none 45.1952%, MAE 6.3296, RMSE 20.0930. The 640px baseline remains exact\n29.8048%, MAE 8.3529. Higher resolution reduces large errors but is not a\nunqualified improvement; do not replace the preserved candidate.\n\nFor 501+ labels per photo, MAE improved 47.2857 -> 26.3377 (about 44%).\nFor 251-500 it improved 12.0544 -> 8.1213. Small-image bands regress.\nThis meets the predeclared rule for one bounded adaptation: at least 20% dense\nMAE reduction without overall MAE regression. Report is in\nruns/pipe_detector_v7_960_diagnostic/report.json; all 1,332 validation images\nwere used, baseline cached scores reproduced checkpoint metrics, no test use.\n\nA dense-batch smoke check at 960px with two 1,523-label training images passed\nforward/backward/optimizer step; peak allocation 8.55 GiB on RTX 5090.\nPlan: pipe_detector_v8_960, initialize v7 best, batch two, head LR 2e-5 and\nbackbone LR 2e-6, max six epochs/patience two, NMS 0.2; new optimizer state.\nKeep original frozen training/validation labels to isolate this experiment;\nthe staged duplicate-label revision is not applied. Compare exact accuracy,\nMAE and density errors before selecting anything. No blind extensions.\n''')
with (workspace/'outputs/Validation-progress.md').open('a',encoding='utf-8') as stream:
    stream.write('''\nWork resumed at the user's request without changing ambiguous labels. A 960px
comparison reduced average error from 8.35 to 6.33 pipes and error on 501+ pipe
photos from 47.3 to 26.3, but exact counts fell from 29.80% to 27.40%.
The existing model remains preserved. One bounded resolution-adaptation run is
planned, capped at six epochs with early stopping, to test whether training at
that resolution can recover exact-count performance. The reserved test remains unused.
''')
print(json.dumps(plan,indent=2))
