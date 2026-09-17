"""Run only after the supervised GPU training slot is free."""
import hashlib, json, time
from pathlib import Path
import torch
from pipe_counter.detection import DetectionDataset, create_detector, configure_detector, pipe_scores
from pipe_counter.fcos import WEIGHTS_NAME, WEIGHTS_SHA256

project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
workspace=Path(__file__).resolve().parents[1]
state=json.loads((project/'runs/supervision/status.json').read_text())
if state.get('phase') in {'starting','training','validation_diagnostic'} or state.get('active_training') or state.get('active_diagnostic'):
    raise RuntimeError('GPU slot is occupied according to supervision status; do not run concurrently.')
torch.set_num_threads(4);torch.manual_seed(42)
torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
torch.hub.set_dir(project/'.cache/torch')
weights=project/'.cache/torch/checkpoints'/WEIGHTS_NAME
assert hashlib.sha256(weights.read_bytes()).hexdigest()==WEIGHTS_SHA256
data=DetectionDataset(project/'data/processed_v2/train.jsonl',geometry_overrides=project/'data/geometry_overrides_v1.json')
assert all(r['split']=='train' for r in data.records)
dense=sorted(range(len(data)),key=lambda i:len(data.records[i]['boxes']),reverse=True)[:2]
empty=next(i for i,r in enumerate(data.records) if not r['boxes'])
small=next(i for i,r in enumerate(data.records) if 1<=len(r['boxes'])<=10)
model=create_detector(960,True,architecture='FCOSResNet50FPN').cuda()
configure_detector(model,.3)
backbone=[p for n,p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]
heads=[p for n,p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
optimizer=torch.optim.AdamW([{'params':backbone,'lr':1e-5},{'params':heads,'lr':1e-4}],weight_decay=.0001,foreach=False)
scaler=torch.amp.GradScaler('cuda');torch.cuda.reset_peak_memory_stats();started=time.monotonic()
batches=[]
for ids in [dense,[small,empty]]:
    samples=[data[i] for i in ids]
    images=[sample[0].cuda() for sample in samples]
    targets=[{k:v.cuda() for k,v in sample[1].items()} for sample in samples]
    model.train();optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):
        losses=model(images,targets);loss=sum(losses.values())
    assert torch.isfinite(loss)
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
    scaler.step(optimizer);scaler.update()
    batches.append({'label_counts':[len(t['boxes']) for t in targets],
        'losses':{k:float(v.detach()) for k,v in losses.items()},'optimizer_step_passed':True})
model.eval()
with torch.inference_mode():
    image,_=data[dense[0]]
    prediction=model([image.cuda()])[0]
    assert torch.isfinite(prediction['scores']).all() and torch.isfinite(prediction['boxes']).all()
    assert len(prediction['scores'])<=2000
    filtered=pipe_scores(prediction)
torch.cuda.synchronize()
result={'architecture':'FCOSResNet50FPN','image_size':960,'device':'cuda','batches':batches,
    'max_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'max_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'elapsed_seconds':time.monotonic()-started,
    'single_photo_float32_inference_passed':True,'candidate_detections':len(filtered),
    'weights_sha256':WEIGHTS_SHA256,'checkpoint_saved':False,'test_evaluated':False,
    'note':'Capacity and numerical smoke check, not training accuracy. Fresh official initialization must be used for the actual experiment.'}
(workspace/'work/fcos_gpu_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
