import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import torch
from pipe_counter.detection import DetectionDataset

project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
geometry=project/'data/geometry_overrides_v1.json'
overrides=json.loads(geometry.read_text())['images']
torch.set_num_threads(1);cv2.setNumThreads(1)
results={};samples={}
for split in ('train','val'):
    dataset=DetectionDataset(project/f'data/processed_v2/{split}.jsonl',geometry_overrides=geometry)
    def check(index):
        image,target=dataset[index]
        assert len(target['boxes'])==len(dataset.records[index]['boxes'])
        assert torch.isfinite(image).all() and torch.isfinite(target['boxes']).all()
        return len(target['boxes'])
    with ThreadPoolExecutor(max_workers=6) as pool:
        counts=list(pool.map(check,range(len(dataset))))
    results[split]={'images':len(counts),'labels':sum(counts),'decoded_and_checked':True}
    affected=[r for r in dataset.records if r['sha256'] in overrides]
    if split=='train':
        chosen=[r for r in affected if r['boxes']][:2]+[r for r in affected if not r['boxes']][:1]
    else:chosen=affected
    chosen += [next(r for r in dataset.records if r['sha256'] not in overrides)]
    samples[split]=chosen
    print(split,results[split],flush=True)
directory=project/'data/smoke_geometry';directory.mkdir(exist_ok=False)
for split,records in samples.items():
    (directory/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
(project/'runs/supervision/geometry_preflight.json').write_text(json.dumps({'results':results,'test_accessed':False},indent=2))
