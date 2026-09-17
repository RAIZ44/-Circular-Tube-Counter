from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import json, hashlib
import cv2
import numpy as np
from pipe_counter.utils import read_jsonl
from pipe_counter.prepare import _clip_box

project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
workspace=Path(__file__).resolve().parents[1]
out=workspace/'outputs/Training-conflict-audit'
out.mkdir(exist_ok=False)
train=read_jsonl(project/'data/processed_v2/train.jsonl')
by_hash={r['sha256']:r for r in train}
conflicts=[c for c in json.loads((project/'data/processed_v2/annotation_conflicts.json').read_text()) if c['sha256'] in by_hash]
source_requests=defaultdict(dict)
variants=defaultdict(list)
for conflict in conflicts:
    for value in conflict['copies']:
        path=Path(value)
        source_requests[path.parent/'_annotations.coco.json'][path.name]=conflict['sha256']
for source,wanted in source_requests.items():
    data=json.loads(source.read_text(encoding='utf-8'))
    cats={int(c['id']) for c in data['categories'] if c['name'].strip().lower() in {'pipe','tube'}}
    records={int(r['id']):r for r in data['images'] if Path(r['file_name']).name in wanted}
    labels=defaultdict(list)
    for a in data['annotations']:
        if int(a['image_id']) in records and int(a['category_id']) in cats:
            r=records[int(a['image_id'])]
            box=_clip_box(a['bbox'],r['width'],r['height'])
            if box is not None:labels[int(a['image_id'])].append(box)
    for i,r in records.items():
        filename=Path(r['file_name']).name
        digest=wanted[filename]
        path=source.parent/filename
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        variants[digest].append({'source_json':str(source),'image':str(path),'source_image_id':i,
            'width':r['width'],'height':r['height'],'count':len(labels[i]),'boxes':labels[i]})
findings=[]
for c in conflicts:
    r=by_hash[c['sha256']]
    vs=variants[c['sha256']]
    counts=sorted(set(v['count'] for v in vs))
    assert counts==c['annotation_counts']
    matches=[v for v in vs if Path(v['image'])==Path(r['image_path'])]
    assert len(matches)==1 and matches[0]['boxes']==r['boxes']
    findings.append({'sha256':c['sha256'],'image':r['image_path'],'split_group':r['split_group'],
        'counts':counts,'spread':max(counts)-min(counts),'retained_count':len(r['boxes']),
        'variants':vs,'all_variant_image_hashes_match':True,'retained_labels_reproduced':True})
findings.sort(key=lambda r:r['spread'],reverse=True)
seen=set();selected=[]
for finding in findings:
    if finding['split_group'] not in seen and finding['spread']>0:
        selected.append(finding);seen.add(finding['split_group'])
    if len(selected)==8:break
for rank,f in enumerate(selected):
    smallest=min(f['variants'],key=lambda v:v['count'])
    biggest=max(f['variants'],key=lambda v:v['count'])
    original=cv2.imread(f['image']);h,w=original.shape[:2]
    scale=min(620/w,760/h);size=(round(w*scale),round(h*scale))
    panels=[]
    for title,variant,color in [('Original',None,(0,0,0)),('Smaller label set',smallest,(255,180,0)),('Retained largest set',biggest,(0,180,0))]:
        frame=cv2.resize(original,size)
        if variant is not None:
            # Refuse silently inventing a geometry mapping during label review.
            assert [variant['width'],variant['height']]==[w,h], f['image']
            for x,y,bw,bh in variant['boxes']:
                cv2.rectangle(frame,(round((x-bw/2)*scale),round((y-bh/2)*scale)),
                    (round((x+bw/2)*scale),round((y+bh/2)*scale)),color,1)
                cv2.circle(frame,(round(x*scale),round(y*scale)),2,color,-1)
        canvas=np.full((820,640,3),245,np.uint8)
        canvas[50:50+size[1],10:10+size[0]]=frame
        suffix=f" {variant['count']}" if variant else ''
        cv2.putText(canvas,f'{rank:02} {title}{suffix}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
        panels.append(canvas)
    cv2.imwrite(str(out/f'{rank:02}.jpg'),np.hstack(panels))
    f['review_panel']=f'{rank:02}.jpg'
report={'created_at':datetime.now(timezone.utc).isoformat(),'split':'train','conflicting_training_images':len(findings),
    'different_count_images':sum(f['spread']>0 for f in findings),
    'variant_copies_verified':sum(len(f['variants']) for f in findings),
    'selection':'Eight largest count discrepancies from distinct training groups, without model predictions. Diagnostic selection only.',
    'training_labels_changed':False,'validation_or_test_used':False,'findings':findings}
(out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='findings'},indent=2))
print(json.dumps([{'panel':f['review_panel'],'counts':f['counts'],'image':f['image']} for f in selected],indent=2))
