import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
workspace=Path(__file__).resolve().parents[1]
outputs=workspace/'outputs'
audit=json.loads((project/'runs/supervision/image_dimensions_audit.json').read_text())
now=datetime.now(timezone.utc).isoformat()
text=f'''
## Detector failure diagnosed and corrected, {now}

pipe_detector_v6 failed during epoch one after at least 100 batches: decoded image
dimensions disagreed with COCO metadata. No epoch/checkpoint was completed. This
was a data-geometry failure, not a GPU failure. The v4 baseline remains preserved.

Decoded all 11,982 training/validation images, excluding reserved test: 20 size
mismatches (17 training, 3 validation). Every affected metadata size was reversed
relative to encoded pixels; ignoring EXIF orientation did not resolve it. Twelve
had nonempty labels. Compared direct coordinates, clockwise and counterclockwise
rotation, and dimension scaling overlays for every labeled mismatch. A 90-degree
counterclockwise label-coordinate transform aligned the boxes in all twelve.
The other eight records retain their empty label sets and use decoded dimensions.
One reviewed scene visibly has partial annotations; spatial alignment does not
prove completeness. This remains a label-quality limitation.

data/geometry_overrides_v1.json is an explicit, SHA256-keyed adapter containing
only the audited images, dimensions, counts and transform. The loader verifies
image bytes and expected geometry; unknown mismatches still fail with a specific
filename. Raw photos, frozen manifests, split groups, and counts were not changed.
The two nonempty affected validation images keep their existing counts, so count
accuracy remains comparable; box-location accuracy would need corrected geometry.
The adapter is included and hashed in detector checkpoints and source snapshots.

Twenty-six tests pass, including transformed box positions, count preservation,
and rejection of unapproved geometry. A full decoded-data preflight and a CUDA
smoke run on corrected and empty records precede restarting the same bounded
detector comparison in pipe_detector_v6_geometry (eight epochs, patience three).
The failed attempt is preserved. No reserved-test data was evaluated or corrected.
'''
with (project/'PRODUCTION_READINESS.md').open('a',encoding='utf-8') as handle:handle.write(text)
shutil.copy2(project/'PRODUCTION_READINESS.md',outputs/'Production-readiness-plan.md')
examples=outputs/'Geometry-review';examples.mkdir(exist_ok=True)
for name in ('00','11','17'):
    shutil.copy2(workspace/f'work/dimension_review/{name}.jpg',examples/f'{name}.jpg')
(outputs/'Geometry-repair.md').write_text('''# Image and annotation alignment repair

The detector's first full run stopped because some image dimensions differed
from their annotation metadata. No completed detector epoch was lost.

All 11,982 training and validation images were checked. Twenty were affected:
17 training and 3 validation. Twelve contained labels; their box coordinates
needed a 90-degree rotation to align with the stored photo. Eight had no labels.
All twelve labeled cases were visually checked using alternative overlays.

The repair uses an explicit per-file correction list and verifies each image's
fingerprint. The original photos, label files, count labels, and split membership
remain unchanged. Unknown mismatches still produce an error for review.

Example sheets in Geometry-review show direct, clockwise, counterclockwise, and
scaled overlays, left to right. The third panel is the confirmed alignment.
This is an AI spatial-alignment review, not verification that every visible pipe
is annotated. Existing annotation completeness issues remain.

Twenty-six regression tests pass. The original best model remains at 28.15%
exact validation counts; the detector has no full validation score yet.
The reserved test set was not inspected in this audit.
''',encoding='utf-8')
path=project/'runs/supervision/experiments.json'
experiments=json.loads(path.read_text(encoding='utf-8-sig'))
experiments['results'].append({'experiment':'pipe_detector_v6','status':'failed_before_first_epoch_complete',
    'reason':'20 decoded-image/annotation dimension mismatches; audited coordinate orientation adapter added',
    'best_checkpoint_written':False})
experiments['next_experiment']['run']='pipe_detector_v6_geometry'
experiments['next_experiment']['status']='prepared'
experiments['next_experiment']['geometry_overrides']='data/geometry_overrides_v1.json'
experiments['updated_at']=now
path.write_text(json.dumps(experiments,indent=2))
report_path=outputs/'Validation-progress.md'
report=report_path.read_text(encoding='utf-8-sig')
report += '\nThe first detector attempt stopped on mismatched image/annotation dimensions.\nTwenty affected files were audited and a documented coordinate correction added.\nThe corrected detector run is being verified and restarted; the baseline is preserved.\n'
report_path.write_text(report,encoding='utf-8')
print('Recorded geometry repair and preserved the failed experiment history.')
