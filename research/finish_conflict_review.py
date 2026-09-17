import json
from pathlib import Path
from datetime import datetime, timezone
workspace=Path(__file__).resolve().parents[1]
project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
now=datetime.now(timezone.utc).isoformat()
out=workspace/'outputs/Training-conflict-audit'
spatial=json.loads((out/'equal_count_spatial_audit.json').read_text())
spatial['visual_review']={'cases_reviewed':4,'selection':'Lowest agreement from four distinct training groups.',
    'finding':'All four reviewed cases mark the same single visible pipe end, with different box tightness/extent. No gross rotation or translation fault was established. This does not certify the remaining records.',
    'labels_changed':False}
(out/'equal_count_spatial_audit.json').write_text(json.dumps(spatial,indent=2),encoding='utf-8')
status_path=project/'runs/supervision/status.json'
status=json.loads(status_path.read_text())
status.update(phase='awaiting_counting_rules',updated_at=now,
    next_action='Await the two counting-rule answers already requested (round-only versus rectangular tubes too; identifiable cropped ends versus complete ends only). Do not repeat questions or launch blind training. After answers, version a concrete annotation policy and plan a bounded audit/experiment; keep frozen validation reference and reserved test unchanged.')
status['counting_rule_review']['questions_asked_once_at']=now
status['counting_rule_review']['equal_count_spatial_audit']=str(out/'equal_count_spatial_audit.json')
status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
exp_path=project/'runs/supervision/experiments.json'
experiments=json.loads(exp_path.read_text())
experiments.update(updated_at=now,next_experiment=None,next_action=status['next_action'])
exp_path.write_text(json.dumps(experiments,indent=2),encoding='utf-8')
with (project/'PRODUCTION_READINESS.md').open('a',encoding='utf-8') as stream:
    stream.write(f'''\nFollow-up spatial audit, {now}: ranked all 33 equal-count training conflicts\nby symmetric best-box IoU and visually inspected the four least-agreeing distinct\ngroups. All four show the same single pipe end with varying box extent/tightness;\nno new gross coordinate transform fault was established. No labels changed.\nTwo concise user questions are now pending about shapes and partial ends.\nAwait these answers before revising ambiguous counting targets. Do not repeat\nthe questions on every heartbeat or launch speculative training while waiting.\nNo active training job; frozen validation and reserved test remain unchanged.\n''')
with (workspace/'outputs/Validation-progress.md').open('a',encoding='utf-8') as stream:
    stream.write('''\nFurther review reconstructed 113 conflicting training-image annotations from
313 source copies. Some differences reflect counting rules: one photo is labeled
as 7 round pipes or 68 ends including rectangular tubes. Others differ on partial
ends or background stacks. [View the examples](Training-conflict-audit/index.html).
Questions about shapes and cropped ends are pending before changing these labels.
Training is currently paused; preserved model accuracy is unchanged.
''')
print(json.dumps({'phase':status['phase'],'questions_pending':2,'training_active':False,'test_evaluated':False}))
