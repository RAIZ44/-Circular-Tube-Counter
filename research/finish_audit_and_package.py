import hashlib, json, copy, shutil
from pathlib import Path
from datetime import datetime, timezone
from pipe_counter.utils import read_jsonl, write_jsonl

workspace=Path(__file__).resolve().parents[1]
project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
now=datetime.now(timezone.utc).isoformat()
audit_dir=workspace/'outputs/Training-duplicate-audit'
audit=json.loads((audit_dir/'audit.json').read_text())
original_manifest=project/'data/processed_v2/train.jsonl'
original=read_jsonl(original_manifest)
assert hashlib.sha256(original_manifest.read_bytes()).hexdigest()==audit['manifest_sha256']
assert audit['all_duplicates_identical_in_raw_source']
by_sha={r['sha256']:r for r in audit['findings']}
revision_dir=project/'data/training_revisions/exact_duplicate_boxes_v1'
revision_dir.mkdir(parents=True,exist_ok=False)
revised=copy.deepcopy(original)
removed=0
for before,after in zip(original,revised):
    if before['sha256'] in by_sha:
        unique=list(dict.fromkeys(tuple(b) for b in before['boxes']))
        assert len(unique)==by_sha[before['sha256']]['unique_box_count']
        after['boxes']=[list(b) for b in unique]
        removed+=len(before['boxes'])-len(unique)
    assert {k:v for k,v in before.items() if k!='boxes'}=={k:v for k,v in after.items() if k!='boxes'}
assert removed==409
assert sum(len(r['boxes']) for r in revised)==1529155
write_jsonl(revision_dir/'train.jsonl',revised)
reloaded=read_jsonl(revision_dir/'train.jsonl')
assert reloaded==revised
assert hashlib.sha256(original_manifest.read_bytes()).hexdigest()==audit['manifest_sha256']
geometry=json.loads((project/'data/geometry_overrides_v1.json').read_text())
overlap=sorted(set(by_sha)&set(geometry['images']))
revision={'created_at':now,'version':1,'operation':'Remove only exact repeated raw-source bounding boxes from training records, preserving the first copy.',
    'original_manifest':str(original_manifest),'original_manifest_sha256':audit['manifest_sha256'],
    'revised_manifest_sha256':hashlib.sha256((revision_dir/'train.jsonl').read_bytes()).hexdigest(),
    'images':len(revised),'affected_images':49,'annotations_before':1529564,'annotations_after':1529155,
    'removed_duplicates':409,'images_and_group_assignments_unchanged':True,
    'source_audit':str(audit_dir/'audit.json'),'source_audit_sha256':hashlib.sha256((audit_dir/'audit.json').read_bytes()).hexdigest(),
    'geometry_override_overlap':overlap,'applied_to_training_run':False,
    'validation_or_test_labels_changed':False,'raw_data_changed':False,'frozen_manifests_changed':False,
    'note':'Versioned training-only revision, staged for a future experiment. Do not use as a standalone manifest directory: validation and test deliberately remain in processed_v2. No accuracy improvement is claimed.'}
(revision_dir/'revision.json').write_text(json.dumps(revision,indent=2),encoding='utf-8')
shutil.copy2(revision_dir/'revision.json',audit_dir/'training_revision.json')
(audit_dir/'README.md').write_text('''# Exact duplicate training-label audit

All 49 affected training images were checked against their original COCO
annotations. The 409 excess boxes are exact duplicates already present in the
exports, not boxes accidentally duplicated by preprocessing or clipping.
All 49 image hashes and reconstructed source annotations matched the manifest.

The largest two records have 203 labels but only 51 distinct boxes; many ends
were labeled four times. The three illustrated cases show repeated boxes in
orange, with their multiplicity. Green boxes occur once.

A separate versioned training manifest removes these exact repeats, preserving
the first copy, every image and every split group. It contains 1,529,155 labels
instead of 1,529,564. Round-trip identity checks passed. Original data, the frozen
split, validation labels and the reserved test are unchanged. This revision has
not been used for a training run and does not imply an accuracy improvement.

The removed labels are only 0.027% of training annotations. This repair does not
explain the full accuracy gap or justify repeated training by itself. Partial-end
conventions, missing annotations and actual model failures still need work.

See audit.json for source annotation IDs, exact repeated coordinates and hashes;
training_revision.json identifies the staged training manifest.
''',encoding='utf-8')

package=workspace/'outputs/Pipe-detector-v7-development'
verification=json.loads((package/'reload_verification.json').read_text())
pin=json.loads((package/'verification_runtime_pin/0000_11.json').read_text())
assert pin['detected_count']==50 and pin['reference_cudnn_allow_tf32'] is True
verification['runtime_pin_check']={'initial_cudnn_allow_tf32':False,'runner_cudnn_allow_tf32':True,
    'runner_cuda_matmul_allow_tf32':False,'fixture':'regression_examples/11.jpg','count':50,'passed':True}
verification['runner_sha256']=hashlib.sha256((package/'scripts/predict_detector.py').read_bytes()).hexdigest()
(package/'reload_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
hashes={str(path.relative_to(package)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(package.rglob('*')) if path.is_file() and '__pycache__' not in path.parts and path.name!='SHA256SUMS.json'}
(package/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
assert all(hashlib.sha256((package/relative).read_bytes()).hexdigest()==digest for relative,digest in hashes.items())

status_path=project/'runs/supervision/status.json'
status=json.loads(status_path.read_text())
assert status['phase']=='data_quality_review' and not status['active_training']
status.update(updated_at=now,last_supervision_check=now,training_label_revision=str(revision_dir/'revision.json'),
    development_package=str(package),package_verification={'cuda_matching_counts':16,'cpu_matching_counts':15,'fixtures':16,
        'cudnn_tf32_reference_pinned':True,'cpu_cuda_parity':False},
    next_action='Continue targeted annotation-conflict and missing-label audit. Exact duplicate training revision is staged; 0.027% label cleanup alone does not justify blind retraining. Any future experiment must explicitly record the revision and retain original validation reference. Keep reserved test unused.')
status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
with (project/'PRODUCTION_READINESS.md').open('a',encoding='utf-8') as stream:
    stream.write(f'''\n## Training-label source audit and v7 packaging, {now}\n\nNo active Python training process was found; v7 logs/history/checkpoints confirm\nnormal five-epoch completion. No additional training was launched.\n\nVerified all 49 training records with 409 repeated boxes against original COCO\nentries and image SHA256 hashes. Every repeated box is identical in the raw\nsource, not a clipping/preparation artifact. Top examples have 203 annotations\nbut 51 distinct boxes, with most ends repeated four times. A separate versioned\ntraining manifest removes exact repeats while preserving images and group IDs:\n{revision_dir / 'train.jsonl'}\nRound-trip verification passed: 10,650 images, 1,529,155 labels. This staged\nrevision has not been used to train. Geometry-override overlap: {len(overlap)}.\nOriginal raw data and frozen manifests remain untouched. This affects only\n0.027% of training labels, so do not present it as the explanation of all errors\nor automatically launch another run solely because it is available.\n\nPackaged v7 with optimizer removed, identical weights/configuration, bundled\ninference source, provenance, metrics and 16 validation regression fixtures:\n{package}\nAll 16 CUDA CLI counts matched prior validation predictions; repeated GPU\ninference also matched. CPU matched 15/16: fixture 11 gives CPU 51 versus CUDA\n50. Diagnostic: cuDNN TF32 enabled yields score 0.42993456 just below threshold\n0.43; disabling TF32 yields 0.43020958 and 51, matching CPU. The package runner\nnow explicitly pins cuDNN TF32 on and CUDA matrix-multiply TF32 off to match the\naudited reference. A test starting with opposite cuDNN setting reproduced 50.\nFloat32 tensor dtype alone does not imply strict full-precision GPU arithmetic.\nCPU count parity is NOT approved; other GPU/runtime combinations remain untested.\nNo weights or thresholds were changed to hide this difference. Local warmed\nGPU forward median was {verification['cuda_warm_inference_ms']['median']:.2f} ms over 16 cases, excluding decode,\ntensor creation, startup and output writing; not an Azure/end-to-end SLA.\nPackage checksum verification passed. This remains a DEVELOPMENT package,\nnot a production release. Test data remains unused.\n''')
with (workspace/'outputs/Validation-progress.md').open('a',encoding='utf-8') as stream:
    stream.write('''\nThe latest detector is now preserved as a [development package](Pipe-detector-v7-development/README.md).
All 16 saved counts reproduced on the RTX 5090. CPU matched 15 of 16; one
near-threshold detection changed the count by one, so CPU/GPU parity is not
approved. The CUDA runner pins the audited arithmetic settings.

The [training-label audit](Training-duplicate-audit/README.md) traced all 409
exact duplicate boxes in 49 training records to the original exports. A separate
training-only revision is staged, with raw files and frozen splits preserved.
It has not been used in training and does not establish an accuracy improvement.
''')
print(json.dumps({'revision':str(revision_dir),'geometry_overlap':len(overlap),'package_files_hashed':len(hashes),
    'cuda_matches':16,'cpu_matches':15,'training_active':False,'test_evaluated':False}))
