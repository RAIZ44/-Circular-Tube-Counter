import hashlib,json
from pathlib import Path
import torch
from pipe_counter.detection import create_detector
from pipe_counter.fcos import WEIGHTS_NAME,WEIGHTS_SHA256
project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
workspace=Path(__file__).resolve().parents[1]
torch.set_num_threads(2);torch.manual_seed(42)
torch.hub.set_dir(project/'.cache/torch')
path=project/'.cache/torch/checkpoints'/WEIGHTS_NAME
assert hashlib.sha256(path.read_bytes()).hexdigest()==WEIGHTS_SHA256
model=create_detector(128,True,architecture='FCOSResNet50FPN').train()
images=[torch.rand(3,128,128),torch.rand(3,128,128)]
targets=[{'boxes':torch.tensor([[20.,20.,40.,40.],[60.,60.,90.,90.]]),'labels':torch.ones(2,dtype=torch.int64)},
    {'boxes':torch.empty((0,4)),'labels':torch.empty(0,dtype=torch.int64)}]
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,foreach=False)
losses=model(images,targets);loss=sum(losses.values());assert torch.isfinite(loss)
loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
result={'architecture':'FCOSResNet50FPN','pretrained_weights_sha256':WEIGHTS_SHA256,
    'official_pretrained_load_passed':True,'positive_and_empty_batch_optimizer_step_passed':True,
    'losses':{k:float(v.detach()) for k,v in losses.items()},'device':'cpu','image_size':128,
    'gpu_dense_smoke_pending':True,'note':'CPU integration check only; no dataset training or accuracy evaluation.'}
(workspace/'work/fcos_cpu_smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
