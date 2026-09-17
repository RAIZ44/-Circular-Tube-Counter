import json
from pathlib import Path
from datetime import datetime, timezone
project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
workspace=Path(__file__).resolve().parents[1]
now=datetime.now(timezone.utc).isoformat()
criteria={'recorded_at':now,'user_request':'you can explore other model architectures as well to improve accuracy. The goal is 95% +',
    'minimum_exact_count_accuracy':.95,'metric':'exact_count_accuracy',
    'metric_definition':'Fraction of photos where predicted count equals the correct count exactly.',
    'interpretation_note':'User supplied 95%+; assistant explicitly stated the working interpretation as at least 95% exactly counted photos, consistent with prior progress reports. This is a target, not measured achievement.',
    'architecture_exploration_authorized':True,'fully_automatic':True,
    'intended_scope':'Side-by-side pipes; nesting outside scope. Preserve existing labels while shape/partial-end choices remain unanswered.',
    'evaluation_policy':'Select architectures/checkpoints/thresholds using frozen grouped validation; preserve reserved test until candidate selection. Verify labels and operating-domain suitability before any production claim.',
    'production_ready':False,'test_evaluated':False,
    'supersedes':'Earlier provisional 99% target. Any older in-flight run status retaining 0.99 is historical and must not override this file.'}
(project/'ACCEPTANCE_CRITERIA.json').write_text(json.dumps(criteria,indent=2),encoding='utf-8')
scope_path=project/'OPERATING_SCOPE.json'
scope=json.loads(scope_path.read_text())
scope.update(release_accuracy_requirement='User requested 95%+; working metric is exact count per photo.',
    minimum_exact_count_accuracy=.95,acceptance_criteria_file='ACCEPTANCE_CRITERIA.json',architecture_exploration_authorized=True)
scope_path.write_text(json.dumps(scope,indent=2),encoding='utf-8')
state_path=project/'runs/supervision/status.json'
state=json.loads(state_path.read_text())
state.pop('provisional_exact_count_target',None)
state.update(updated_at=now,minimum_exact_count_accuracy=.95,acceptance_criteria_file=str(project/'ACCEPTANCE_CRITERIA.json'),
    operating_scope=scope,active_training=state['phase']=='training')
state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
smoke=json.loads((workspace/'work/fcos_cpu_smoke.json').read_text())
queue={'created_at':now,'policy':'One GPU training job at a time. Do not interrupt v8_960 or replace existing checkpoints. Check live process and lock before launch. Complete dense-batch CUDA preflight after the GPU becomes free.',
    'next':{'architecture':'FCOSResNet50FPN','run_name':'pipe_fcos_v1_960','status':'prepared_waiting_for_gpu',
        'image_size':960,'batch_size':2,'epochs_max':12,'patience':4,'learning_rate':1e-4,
        'box_nms_threshold':.3,'pretrained':True,'initial_checkpoint':None,
        'max_detections':2000,'topk_candidates':10000,'foreground_label':1,
        'data_policy':'Unchanged processed_v2 train and validation, with existing geometry adapter. Do not use test. Do not silently apply the staged label revision.',
        'hypothesis':'One-stage anchor-free detection removes the separate region-proposal sampling stage; test whether this improves dense and small-end counting.',
        'checkpoint_selection':'Validation exact-count accuracy, MAE tie-break; separately report dense-scene error.',
        'preparation_checks':{'cpu_suite_passed':28,'offline_reload_exact_output_parity':True,'pretrained_cpu_smoke':smoke,'gpu_dense_smoke_passed':False},
        'launch_arguments':['--run-name','pipe_fcos_v1_960','--image-size','960','--batch-size','2','--epochs','12','--patience','4','--learning-rate','0.0001','--architecture','FCOSResNet50FPN','--pretrained','--box-nms-threshold','0.3','--selection-metric','exact_count_accuracy']},
    'later_research':[{'architecture':'RF-DETR Small or Large','status':'research_only',
        'constraint':'Default 300 object queries/outputs is below dense image counts. Both num_queries and num_select must be raised and fine-tuned, or use a carefully validated tile-and-merge pipeline. Benchmark capacity and memory before committing.',
        'environment_policy':'Prepare a separate environment; do not upgrade packages in the active training environment.',
        'sources':['https://github.com/roboflow/rf-detr/blob/develop/docs/faq.md','https://github.com/roboflow/rf-detr/blob/develop/docs/index.md']},
        {'architecture':'YOLO26 with small-object head','status':'researched_alternative',
         'constraint':'Official P2 architecture is supplied as YAML, not a pretrained P2 checkpoint. Validate dense output cap and pretrained transfer before comparing. Record official license terms before choosing production packaging.',
         'source':'https://docs.ultralytics.com/models/yolo26'}]}
(project/'runs/supervision/architecture_queue.json').write_text(json.dumps(queue,indent=2),encoding='utf-8')
with (project/'PRODUCTION_READINESS.md').open('a',encoding='utf-8') as stream:
    stream.write(f'''\n## 95% target and architecture exploration, {now}\n\nUser explicitly authorized other architectures and set a 95%+ goal. The working\nmetric stated to the user is at least 95% exactly counted photos. This replaces\nthe older provisional 99% target; ACCEPTANCE_CRITERIA.json is authoritative.\nThe already-running v8 supervisor loaded older metadata at startup and may write\na stale 99% field when it exits; reconcile that with the criteria file rather\nthan interpreting it as a user change. Future supervisor writes load the new file.\n\nV8 continues; no second GPU job launched. Prepared FCOSResNet50FPN as the next\narchitecture, using cached official COCO initialization with verified SHA256\n{smoke['pretrained_weights_sha256']}.\nFCOS retains pretrained feature/box heads, replaces only the final class logits,\nuses frozen batch-norm consistently for pretrained and offline construction,\npermits 2,000 detections/10,000 candidates per level, and filters class 0 from\ncounts. Generic detector train/reload/inference and supervisor now support it.\nFuture training records explicit cuDNN/matmul TF32 settings; inference honors\nthese recorded fields. Existing v8 process and packaged candidates are unchanged.\n\n28 CPU tests pass, including exact offline FCOS reload outputs and rejection of\ncross-architecture initialization. Official-weight load plus a positive/empty\nCPU training batch and optimizer step passed. GPU dense-batch preflight is\ndeferred until v8 releases the GPU. Queue and bounded launch settings are in\nruns/supervision/architecture_queue.json: 960px, batch two, LR 1e-4, max 12\nepochs, patience four, NMS .3, original frozen train/validation labels.\n\nRF-DETR is a later transformer candidate, but the documented 300-query defaults\nwould cap dense counts. Raising query/output capacity requires adaptation and\nmore memory; tiling needs boundary/duplicate handling. YOLO26 was also researched\nas an alternative. No benchmark mAP is substituted for exact-count accuracy,\nno package changes were made to the active environment, and no production result\nis implied by architecture choice. Test remains reserved.\n''')
(workspace/'outputs/Architecture-comparison.md').write_text('''# Architecture comparison and accuracy target

The working target is **at least 95% of photos counted exactly right**. The
current best exact-count validation result is 29.88%; this is a target to work
toward, not a promise that the existing data can establish production accuracy.
The reserved test remains unused.

| Candidate | Status | Reason to evaluate |
|---|---|---|
| Higher-resolution Faster R-CNN | Training now, at most six epochs | Larger input reduced dense-scene errors; adaptation may recover exact counts |
| FCOS ResNet50 FPN | Implemented and CPU-tested; queued after the active run | Different, single-stage detection method; compare count errors on the same split |
| RF-DETR | Research candidate | Transformer-based detection; must first address dense-scene object limits |
| YOLO26 / small-object variant | Researched alternative | Another detector family; requires separate training and deployment review |

FCOS has 2,000 output slots and a larger candidate budget for this experiment.
Its official pretrained weights loaded successfully; 28 CPU tests passed,
including exact offline reload checks. A CPU training step with positive and
empty examples passed. Full-resolution GPU preflight remains pending until the
current run finishes. The planned comparison is capped at 12 epochs with early
stopping; it has not started or demonstrated an accuracy improvement yet.

RF-DETR defaults to 300 queries for core models. Its documentation requires
increasing both query and output counts to handle more objects, followed by
fine-tuning. This must be addressed for photos containing hundreds or thousands
of pipes. [Official RF-DETR FAQ](https://github.com/roboflow/rf-detr/blob/develop/docs/faq.md)

FCOS is a one-stage, anchor-free detector.
[Official Torchvision FCOS documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fcos_resnet50_fpn.html)

YOLO26 supplies a small-object P2 architecture as YAML; pretrained transfer and
dense output limits need validation before a comparison.
[Official YOLO26 documentation](https://docs.ultralytics.com/models/yolo26)

All comparisons retain grouped train/validation membership. Original images and
labels are preserved. Architectural improvements must be measured alongside
label quality and performance on the intended side-by-side setup.
''',encoding='utf-8')
with (workspace/'outputs/Validation-progress.md').open('a',encoding='utf-8') as stream:
    stream.write('''\nThe user set a 95%+ target and authorized other architectures. The working
metric is exactly correct photos. FCOS is implemented and CPU-tested, queued
after the active high-resolution run; RF-DETR and YOLO are additional researched
options. See [the architecture comparison](Architecture-comparison.md).
''')
print(json.dumps({'target':.95,'metric':'exact_count_accuracy','queued_architecture':'FCOSResNet50FPN','gpu_training_started':False,'cpu_tests_passed':28}))
