import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

project = Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
outputs = Path(__file__).resolve().parents[1]/'outputs'
now = datetime.now(timezone.utc).isoformat()
scope = {'recorded_at':now,'user_clarification':'well in my situation, that wont be a problem. The pictures are always going to be beside each other',
    'interpretation':'Pipes arranged side by side; nested pipes are outside the intended operating case.',
    'nested_count_rule_required':False,'fully_automatic':True,'future_host':'Azure',
    'release_accuracy_requirement':'Not yet quantified by user; existing 99% exact-count target remains provisional.',
    'evaluation_policy':'Keep the full frozen validation split for comparable experiment results. Do not remove difficult images based on model errors or relabel nested examples as outer-only.'}
(project/'OPERATING_SCOPE.json').write_text(json.dumps(scope,indent=2),encoding='utf-8')
status_path = project/'runs/supervision/status.json'
state = json.loads(status_path.read_text())
assert state['phase']=='awaiting_count_definition'
shutil.copy2(status_path,project/'runs/supervision/v4_before_scope_resume.json')
state.update(phase='preparing_crop_experiment',updated_at=now,operating_scope=scope,
    next_action='Run the bounded crop-augmented fine-tune, compare full-image validation with preserved 28.15% exact / 9.33 MAE candidate. Inspect density errors; test tiled inference only after crop adaptation and only on validation. Nested-pipe rules are no longer a blocker for the user scope.')
state.pop('awaiting_user_decision',None)
state.pop('blocker_last_reported_at',None)
temp=status_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(status_path)
experiments_path=project/'runs/supervision/experiments.json'
experiments=json.loads(experiments_path.read_text())
experiments.update(updated_at=now,next_action=state['next_action'],next_experiment={
    'run':'pipe_center_v5_crop_exact','status':'prepared',
    'hypothesis':'Dense stacks lose detail at full-image 512px scaling. Training half the views as 50-100% crops should teach scale variation and improve small-end recognition without abandoning full-photo context.',
    'initial_checkpoint':'runs/pipe_center_v4_float32/candidate.pt','architecture':'ResNet18CenterNet',
    'image_size':512,'batch_size':16,'epochs_max':15,'patience':5,'learning_rate':0.00005,
    'crop_probability':0.5,'crop_min_scale':0.5,'validation_precision':'float32','validation_batch_size':1,
    'baseline_exact_accuracy':0.28153153153153154,'baseline_mae':9.332582582582583,
    'labels':'No label changes; crops retain centers inside half-open bounds and translate coordinates. Original manifests and split memberships remain frozen.',
    'decision_rule':'Retain only if validation improves. If it fails, preserve baseline, analyze specific remaining errors and do not endlessly repeat crop or resolution variants.'})
experiments_path.write_text(json.dumps(experiments,indent=2))
update=f'''
## Side-by-side operating scope and resumed work, {now}

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
'''
with (project/'PRODUCTION_READINESS.md').open('a',encoding='utf-8') as handle:handle.write(update)
shutil.copy2(project/'PRODUCTION_READINESS.md',outputs/'Production-readiness-plan.md')
report=outputs/'Validation-progress.md'
text=report.read_text()
text=text.replace('The unresolved rule is whether smaller pipes nested\ninside larger pipes count too. No labels were silently corrected. Label-dependent\ntraining is waiting for that decision; no full training job is currently active.\nMonitoring remains active and will stay quiet while the decision is unchanged.',
    'The user clarified that pipes will be side by side, so nested pipes are outside\nthe intended operating case. The counting-rule blocker is resolved. No labels\nwere silently corrected. A bounded crop-augmented fine-tune is being started to\nimprove recognition of small ends in dense stacks. Monitoring remains active.')
report.write_text(text,encoding='utf-8')
print('Recorded scope, cleared blocker, and documented the bounded crop experiment.')
