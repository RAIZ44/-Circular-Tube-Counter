import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import torch
from pipe_counter.detection import load_detector,DetectionDataset

project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
report_path=project/'runs/validation_diagnostics/v6_nms/report.json'
report=json.loads(report_path.read_text())
parent_path=project/'runs/pipe_detector_v6_geometry/best.pt'
checkpoint=torch.load(parent_path,map_location='cpu',weights_only=False)
checkpoint.pop('optimizer',None)
checkpoint['original_metrics']=checkpoint['metrics']
checkpoint['metrics']=report['variants']['0.2']
checkpoint['threshold']=checkpoint['metrics']['threshold']
checkpoint['config']['box_nms_threshold']=.2
checkpoint['calibration']={'time':datetime.now(timezone.utc).isoformat(),
    'parent_sha256':hashlib.sha256(parent_path.read_bytes()).hexdigest(),
    'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
    'report':str(report_path),'test_evaluated':False,'production_ready':False,
    'note':'Validation-selected NMS and threshold; weights unchanged. Exact accuracy improves but MAE remains worse than the heatmap baseline.'}
destination=project/'runs/pipe_detector_v6_calibrated';destination.mkdir(exist_ok=False)
torch.save(checkpoint,destination/'candidate.pt')
(destination/'calibration.json').write_text(json.dumps(checkpoint['calibration'],indent=2))
torch.set_num_threads(4)
model,loaded=load_detector(destination/'candidate.pt',torch.device('cuda'))
assert model.roi_heads.nms_thresh==.2
dataset=DetectionDataset(project/'data/processed_v2/val.jsonl',geometry_overrides=project/'data/geometry_overrides_v1.json')
expected=list(csv.DictReader((report_path.parent/'per_image_iou_0.2.csv').open()))
selected=list(range(12))+[219,469,852,1288]
rows=[]
with torch.inference_mode():
    for index in selected:
        values=model([dataset[index][0].cuda()])[0]['scores']
        count=int((values>=loaded['threshold']).sum())
        rows.append({'index':index,'expected':int(expected[index]['predicted_count']),'actual':count})
        assert rows[-1]['expected']==count,rows[-1]
(destination/'reload_verification.json').write_text(json.dumps({'images':len(rows),'matches':len(rows),'cases':rows},indent=2))
print({'candidate':str(destination/'candidate.pt'),'verified_images':len(rows),'metrics':checkpoint['metrics']})
