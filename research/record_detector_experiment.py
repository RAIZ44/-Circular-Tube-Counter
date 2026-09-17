import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
outputs=Path(__file__).resolve().parents[1]/'outputs'
supervision=project/'runs/supervision'
state=json.loads((supervision/'status.json').read_text())
assert state['phase']=='validation_review' and state['training_exit_code']==0 and state['run_dir'].endswith('pipe_center_v5_crop_exact')
experiments=json.loads((supervision/'experiments.json').read_text(encoding='utf-8-sig'))
experiments['updated_at']=datetime.now(timezone.utc).isoformat()
for name,path in [('v5_best','runs/validation_diagnostics/v5_best/512/report.json'),('v5_tiles','runs/validation_diagnostics/v5_tiles/report.json')]:
    result=json.loads((project/path).read_text())
    experiments['results'].append({'experiment':name,'metrics':result['optimized_for_exact_count'],'report':str(project/path),
        'decision':'Rejected: lower exact accuracy than preserved v4 float32 baseline.'})
experiments['next_experiment']={'run':'pipe_detector_v6','status':'prepared','architecture':'FasterRCNNResNet50FPNV2',
    'hypothesis':'Direct object/background classification and box-boundary regression supply supervision missing from the center-only model; pretrained multi-scale features may separate crowded ends.',
    'initialization':'Official COCO pretrained torchvision Faster R-CNN ResNet50 FPN V2',
    'image_size':640,'batch_size':2,'learning_rate':.0002,'backbone_lr':.00002,'epochs_max':8,'patience':3,
    'inference_precision':'float32','validation_batch_size':1,'nms_iou':.5,'max_detections':2000,
    'anchor_sizes':[16,32,64,128,256],'selection_metric':'exact_count_accuracy',
    'baseline_exact_accuracy':.28153153153153154,'baseline_mae':9.332582582582583,
    'label_policy':'Original frozen train/validation records and counts; convert center/size to clipped corners, rejecting invalid boxes rather than dropping labels.',
    'stop_policy':'At most eight epochs with patience three. Preserve the prior candidate unless validation exact counts improve. Reject failed architecture comparison rather than endlessly increasing epochs.'}
experiments['next_action']='Train and evaluate the bounded box-detector comparison. Keep test reserved. Side-by-side operating scope remains in effect; nested pipes are not a blocker.'
(supervision/'experiments.json').write_text(json.dumps(experiments,indent=2),encoding='utf-8')
update=f'''
## Crop rejection and box-detector comparison, {experiments['updated_at']}

The crop experiment stopped normally after six epochs (best epoch one): 25.38%
exact counts, MAE 8.90. Full validation diagnostics reproduced its recorded score.
Context-tile inference after crop adaptation reached only 14.26% exact, MAE 10.66.
Reject both for the primary exact-count objective. Preserve v4_float32 at 28.15%
exact / MAE 9.33. Do not repeat crop/resolution variants without new evidence.

The next distinct bounded comparison uses box supervision already present in
the labels, rather than only Gaussian center targets. Torchvision Faster R-CNN
ResNet50 FPN V2 starts from the official COCO weights. It learns pipe/background
classification and box regression with features at multiple spatial scales.
Configuration: 640px maximum image side, batch 2, AdamW LR 0.0002 for heads and
0.00002 for backbone, up to eight epochs, patience three, float32 batch-one
validation, exact-count threshold/checkpoint selection. Early pretrained ResNet
layers are frozen; BatchNorm running statistics are fixed for small batches.

Dense-scene settings explicitly replace the default 100 detections: 2,000 final
boxes, 4,000 post-NMS proposals at inference, anchors 16/32/64/128/256 pixels,
box NMS IoU 0.5. Training labels have at most 1,523 boxes; validation at most 1,121.
At 640px, minimum box-side percentiles are 7.85/15/23/32.62/60.8 at 5/25/50/75/95%.
All train/validation boxes passed conversion checks; no labels were removed.
No reserved-test images or labels were loaded for this experiment.

Twenty-five tests passed. CUDA smoke training succeeded at 128px and at 640px
on two images with 1,523 labels each. The latter checkpoint reloaded offline and
reproduced both validation fixture counts. Smoke scores are not accuracy evidence.
Official weights SHA256: dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf.
Reference: https://docs.pytorch.org/vision/0.26/_modules/torchvision/models/detection/faster_rcnn.html
New detector checkpoints use scripts/predict_detector.py, not the center-heatmap
predict.py. The saved v4 development package remains unchanged. Compare full frozen
validation before promoting the detector; do not evaluate reserved test merely
because training finishes. No production-readiness claim is warranted yet.
'''
with (project/'PRODUCTION_READINESS.md').open('a',encoding='utf-8') as f:f.write(update)
shutil.copy2(project/'PRODUCTION_READINESS.md',outputs/'Production-readiness-plan.md')
report=outputs/'Validation-progress.md'
text=report.read_text(encoding='utf-8-sig')
text=text.replace('A bounded crop-augmented fine-tune is running on the RTX 5090 to\nimprove recognition of small ends in dense stacks. Monitoring remains active.',
    'The crop-trained experiment finished at 25.38% exact and section-counting at\n14.26%; both were rejected. A bounded box-detector comparison is being started\nto use the existing object outlines as additional supervision. Monitoring remains active.')
report.write_text(text,encoding='utf-8')
print('Recorded rejected crop approaches and the bounded detector comparison.')
