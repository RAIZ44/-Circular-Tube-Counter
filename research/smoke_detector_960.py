import json, time
from pathlib import Path
import torch
from pipe_counter.detection import DetectionDataset, create_detector
project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
workspace=Path(__file__).resolve().parents[1]
torch.set_num_threads(4);torch.manual_seed(42)
torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
data=DetectionDataset(project/'data/processed_v2/train.jsonl',geometry_overrides=project/'data/geometry_overrides_v1.json')
ids=sorted(range(len(data)),key=lambda i:len(data.records[i]['boxes']),reverse=True)[:2]
samples=[data[i] for i in ids]
checkpoint=torch.load(project/'runs/pipe_detector_v7_finetune/best.pt',map_location='cpu',weights_only=False)
model=create_detector(960).cuda();model.load_state_dict(checkpoint['model']);del checkpoint
model.roi_heads.nms_thresh=.2;model.train()
for module in model.modules():
    if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
backbone=[p for n,p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]
heads=[p for n,p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
optimizer=torch.optim.AdamW([{'params':backbone,'lr':2e-6},{'params':heads,'lr':2e-5}],weight_decay=.0001,foreach=False)
scaler=torch.amp.GradScaler('cuda');torch.cuda.reset_peak_memory_stats();started=time.monotonic()
images=[s[0].cuda() for s in samples]
targets=[{k:v.cuda() for k,v in s[1].items()} for s in samples]
with torch.autocast('cuda',dtype=torch.float16):
    losses=model(images,targets);loss=sum(losses.values())
assert torch.isfinite(loss)
scaler.scale(loss).backward();scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(),10.);scaler.step(optimizer);scaler.update()
torch.cuda.synchronize()
result={'image_size':960,'batch_size':2,'training_labels':[len(t['boxes']) for t in targets],
    'loss':float(loss.detach()),'finite_loss_and_optimizer_step':True,
    'max_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'max_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'elapsed_seconds':time.monotonic()-started,
    'checkpoint_saved':False,'raw_data_changed':False,'test_evaluated':False}
(workspace/'work/detector_960_smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
