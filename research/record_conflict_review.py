import json, html
from pathlib import Path
from datetime import datetime, timezone
workspace=Path(__file__).resolve().parents[1]
project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
out=workspace/'outputs/Training-conflict-audit'
report=json.loads((out/'audit.json').read_text())
notes=[
    'The largest difference includes square/rectangular tube ends on the lower shelves. The smaller set omits these while labeling round ends. This reflects a target-class convention, not simply a missed detection.',
    'The seven-label version marks round ends at right; the 68-label version also marks the square/rectangular tubes. Identical pixels have different intended counting targets.',
    'The larger set adds partly obscured or recessed ends between rows. Their inclusion requires careful visibility review; neither export count is automatically certified here.',
    'The larger set includes a separate small background stack as well as the foreground pipes. Foreground-only versus every visible stack changes the target count.',
    'The larger set covers additional ends, including image-boundary pieces. The smaller set also leaves some clearly visible ends unmarked.',
    'Much of the difference comes from partial ends near image edges. Some fully visible ends also appear absent from the smaller set.',
    'Additional visible ends appear in the larger set, especially lower and right areas. This demonstrates incomplete annotations in at least one version; the precise complete count has not been independently certified.',
    'The larger set includes more partially visible ends at the left, bottom and right boundaries. The choice to count partial ends needs to be consistent.'
]
selected=sorted([r for r in report['findings'] if 'review_panel' in r],key=lambda r:r['review_panel'])
cards=[]
for row,note in zip(selected,notes):
    row['visual_observation']=note
    row['observation_status']='AI visual comparison only; no automatic relabeling or certified count.'
    cards.append(f'<section><h2>Case {row["review_panel"][:2]} — export counts {", ".join(str(c) for c in row["counts"])}</h2><p>{html.escape(note)}</p><a href="{row["review_panel"]}"><img src="{row["review_panel"]}" alt="Original and two conflicting annotation sets"></a></section>')
report['visual_reviewed_at']=datetime.now(timezone.utc).isoformat()
report['unresolved_counting_rules']=['Round pipes only or round plus square/rectangular tubes','Whether identifiable ends cut by image borders count','Whether separate background stacks count']
(out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Training annotation conflicts</title><style>body{font:17px system-ui;background:#f5f7fa;color:#18212d;max-width:1500px;margin:32px auto;padding:0 20px}section{background:white;margin:24px 0;padding:20px;border-radius:12px}img{width:100%;height:auto}p{line-height:1.5;max-width:1000px}</style><h1>What should count as a pipe?</h1><p>113 training images have conflicting export annotations; 80 have different counts. All 313 corresponding image copies and retained annotations were checked against source files. These eight diagnostic cases were selected by count disagreement, without model predictions. They are not an accuracy sample.</p><p>Some conflicts come from different definitions: round versus rectangular tubing, partially visible ends and separate background stacks. Other labels are simply incomplete. Choosing the largest label set does not settle the intended counting rule. Nesting is already outside the intended use. No ambiguous labels, validation records or reserved-test records were changed.</p>''' + ''.join(cards)+'</html>',encoding='utf-8')
status_path=project/'runs/supervision/status.json'
status=json.loads(status_path.read_text())
status.update(updated_at=report['visual_reviewed_at'],last_supervision_check=report['visual_reviewed_at'],
    training_conflict_audit=str(out/'index.html'),
    counting_rule_review={'training_images_with_conflicts':113,'training_images_with_count_conflicts':80,
        'verified_export_copies':313,'reviewed_cases':8,'shape_and_boundary_questions_pending':True},
    next_action='Resolve counting semantics for round versus rectangular tubes and image-boundary ends before modifying ambiguous labels. Continue independent geometry/localization audit of equal-count annotation variants; no blind retraining or test-set tuning.')
status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
with (project/'PRODUCTION_READINESS.md').open('a',encoding='utf-8') as stream:
    stream.write(f'''\n## Training annotation-conflict review, {report['visual_reviewed_at']}\n\nReconstructed every training conflict from source COCO entries: 113 retained\ntraining images, 80 with differing counts, 313 source image copies hash-verified.\nEvery selected training label list reproduced exactly. Eight largest differences\nfrom distinct groups were visually reviewed without model predictions.\n\nThe 7-versus-68 and 360-versus-531 examples expose different target definitions:\nround ends only versus round plus square/rectangular tubes. Other examples differ\non partially cropped ends and separate background stacks. Some sets also omit\nclearly visible ends. Max-annotation selection is not equivalent to verified truth.\nDo not silently switch counts or declare either export correct. Nesting remains\nout of scope, as the user already clarified. Illustrated evidence:\n{out / 'index.html'}\n\nAsk the user to define shape and boundary treatment once, with concrete examples.\nNo ambiguous annotations changed; original frozen train/val/test maintained.\nWhile waiting, equal-count variants can still be audited for spatial errors\nwithout guessing the user's counting target. No additional training launched.\n''')
print(out/'index.html')
