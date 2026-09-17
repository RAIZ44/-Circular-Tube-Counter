import json
from pathlib import Path
import cv2
import numpy as np
workspace=Path(__file__).resolve().parents[1]
out=workspace/'outputs/Training-conflict-audit'
report=json.loads((out/'audit.json').read_text())
def xyxy(boxes):
    b=np.asarray(boxes,dtype=np.float64)
    return np.concatenate([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2],axis=1)
def overlap(a,b):
    a=xyxy(a);b=xyxy(b)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    intersection=np.maximum(hi-lo,0).prod(axis=2)
    area_a=(a[:,2:]-a[:,:2]).prod(axis=1)
    area_b=(b[:,2:]-b[:,:2]).prod(axis=1)
    iou=intersection/(area_a[:,None]+area_b[None,:]-intersection+1e-9)
    return float((iou.max(axis=0).mean()+iou.max(axis=1).mean())/2)
pairs=[]
for row in report['findings']:
    if len(row['counts'])!=1 or row['counts'][0]==0:continue
    variants=row['variants']
    worst=None
    for i,a in enumerate(variants):
        for j,b in enumerate(variants[:i]):
            if sorted(a['boxes'])==sorted(b['boxes']):continue
            score=overlap(a['boxes'],b['boxes'])
            candidate={'sha256':row['sha256'],'image':row['image'],'split_group':row['split_group'],
                'count':a['count'],'variant_indices':[i,j],'mean_bidirectional_best_iou':score,
                'metadata_sizes':[[a['width'],a['height']],[b['width'],b['height']]]}
            if worst is None or score<worst['mean_bidirectional_best_iou']:worst=candidate
    if worst:pairs.append(worst)
pairs.sort(key=lambda row:row['mean_bidirectional_best_iou'])
by_sha={r['sha256']:r for r in report['findings']}
seen=set();selected=[]
for row in pairs:
    if row['split_group'] not in seen:
        selected.append(row);seen.add(row['split_group'])
    if len(selected)==4:break
for rank,row in enumerate(selected):
    original=cv2.imread(row['image']);h,w=original.shape[:2]
    variants=[by_sha[row['sha256']]['variants'][i] for i in row['variant_indices']]
    scale=min(620/w,760/h);size=(round(w*scale),round(h*scale));panels=[]
    for title,variant,color in [('Original',None,(0,0,0)),('Annotation A',variants[0],(255,180,0)),('Annotation B',variants[1],(0,180,0))]:
        frame=cv2.resize(original,size)
        if variant:
            assert [variant['width'],variant['height']]==[w,h]
            for x,y,bw,bh in variant['boxes']:
                cv2.rectangle(frame,(round((x-bw/2)*scale),round((y-bh/2)*scale)),(round((x+bw/2)*scale),round((y+bh/2)*scale)),color,1)
                cv2.circle(frame,(round(x*scale),round(y*scale)),2,color,-1)
        canvas=np.full((820,640,3),245,np.uint8);canvas[50:50+size[1],10:10+size[0]]=frame
        cv2.putText(canvas,f'{rank:02} {title} ({row["count"]} labels)',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
        panels.append(canvas)
    filename=f'spatial_{rank:02}.jpg';cv2.imwrite(str(out/filename),np.hstack(panels));row['panel']=filename
(out/'equal_count_spatial_audit.json').write_text(json.dumps({'method':'Ranking only: symmetric mean best-box IoU, not one-to-one label matching or correctness certification.','pairs':pairs},indent=2),encoding='utf-8')
print(json.dumps({'records':len(pairs),'selected':selected},indent=2))
