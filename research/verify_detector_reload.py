import csv
from pathlib import Path
import torch
from pipe_counter.detection import load_detector,DetectionDataset

project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
torch.set_num_threads(4)
model,checkpoint=load_detector(project/'runs/repair_detector_dense_smoke/best.pt',torch.device('cuda'))
dataset=DetectionDataset(project/'data/smoke_detector_dense/val.jsonl')
expected=[int(r['predicted_count']) for r in csv.DictReader((project/'runs/repair_detector_dense_smoke/best_validation_per_image.csv').open())]
with torch.inference_mode():
    actual=[int((model([dataset[i][0].cuda()])[0]['scores']>=checkpoint['threshold']).sum()) for i in range(len(dataset))]
print({'expected':expected,'actual':actual})
assert actual==expected
