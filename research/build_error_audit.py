import csv
import json
import random
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import torch
from pipe_counter.detection import DetectionDataset,load_detector
from pipe_counter.utils import read_jsonl

project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
workspace=Path(__file__).resolve().parents[1]
out=workspace/'work/v7_error_audit';out.mkdir(exist_ok=False)
torch.set_num_threads(4)
data=DetectionDataset(project/'data/processed_v2/val.jsonl',geometry_overrides=project/'data/geometry_overrides_v1.json')
rows=list(csv.DictReader((project/'runs/pipe_detector_v7_finetune/best_validation_per_image.csv').open()))
eligible=[i for i,r in enumerate(data.records) if 1<=len(r['boxes'])<=30 and int(rows[i]['error'])!=0 and '4-train.' not in r['source_json']]
random.Random(42).shuffle(eligible)
selected=[0];groups={data.records[0]['split_group']}
for i in eligible:
    if data.records[i]['split_group'] not in groups:
        selected.append(i);groups.add(data.records[i]['split_group'])
    if len(selected)==10:break
model,checkpoint=load_detector(project/'runs/pipe_detector_v7_finetune/best.pt',torch.device('cuda'))
cases=[]
with torch.inference_mode():
    for j,i in enumerate(selected):
        image,target=data[i];r=data.records[i]
        pred=model([image.cuda()])[0];keep=pred['scores']>=checkpoint['threshold']
        predicted=pred['boxes'][keep].cpu().numpy()
        original=cv2.cvtColor((image.permute(1,2,0).numpy()*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
        h,w=original.shape[:2];s=min(620/w,600/h);size=(round(w*s),round(h*s))
        panels=[]
        for name,boxes,color in [('Original',[],(0,0,0)),('Labels',target['boxes'].numpy(),(0,255,0)),('Model',predicted,(0,165,255))]:
            frame=cv2.resize(original,size)
            for box in boxes:
                x1,y1,x2,y2=box*s
                cv2.rectangle(frame,(round(x1),round(y1)),(round(x2),round(y2)),color,1)
                cv2.circle(frame,(round((x1+x2)/2),round((y1+y2)/2)),3,color,-1)
            canvas=np.full((650,640,3),245,np.uint8)
            canvas[50:50+size[1],10:10+size[0]]=frame
            cv2.putText(canvas,f'{j:02d} {name}'+(f' {len(boxes)}' if name!='Original' else ''),(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
            panels.append(canvas)
        path=out/f'{j:02d}.jpg';cv2.imwrite(str(path),np.hstack(panels))
        cases.append({'case':j,'index':i,'image':r['image_path'],'sha256':r['sha256'],'labels':len(r['boxes']),
            'predicted':len(predicted),'split_group':r['split_group'],'panel':str(path)})
train=read_jsonl(project/'data/processed_v2/train.jsonl')
duplicate=[]
for r in train:
    counts=Counter(tuple(b) for b in r['boxes']);excess=sum(n-1 for n in counts.values())
    if excess:duplicate.append({'image':r['image_path'],'labels':len(r['boxes']),'duplicate_excess':excess,'sha256':r['sha256']})
duplicate.sort(key=lambda r:r['duplicate_excess'],reverse=True)
(out/'index.json').write_text(json.dumps({'selection':'Diagnostic error cases, not an accuracy sample; 10 distinct validation groups with <=30 labels from exports 1-3. Seed 42 after a fixed first example.',
    'cases':cases,'training_exact_duplicate_boxes':duplicate},indent=2))
print(json.dumps({'cases':cases,'largest_duplicate_examples':duplicate[:5]},indent=2))
