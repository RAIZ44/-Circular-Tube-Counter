"""Trace exact repeated training boxes to their original COCO entries; no edits."""
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import cv2
import numpy as np
from pipe_counter.utils import read_jsonl
from pipe_counter.prepare import _clip_box

root = Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
workspace = Path(__file__).resolve().parents[1]
out = workspace / 'outputs/Training-duplicate-audit'
out.mkdir(exist_ok=False)
manifest = root / 'data/processed_v2/train.jsonl'
before_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
records = read_jsonl(manifest)
affected = []
sources = defaultdict(list)
for record in records:
    counts = Counter(tuple(box) for box in record['boxes'])
    if len(counts) < len(record['boxes']):
        affected.append(record)
        sources[record['source_json']].append(record)

findings = []
for source, group in sources.items():
    payload = json.loads(Path(source).read_text(encoding='utf-8'))
    categories = {int(c['id']):c['name'].strip().lower() for c in payload['categories']}
    wanted = {r['source_image_id'] for r in group}
    indexed = defaultdict(list)
    for annotation in payload['annotations']:
        if annotation['image_id'] in wanted and categories[int(annotation['category_id'])] in {'pipe','tube'}:
            indexed[annotation['image_id']].append(annotation)
    source_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    for record in group:
        annotations = indexed[record['source_image_id']]
        reproduced = []
        raw_by_converted = defaultdict(list)
        for annotation in annotations:
            converted = _clip_box(annotation['bbox'],record['width'],record['height'])
            if converted is not None:
                reproduced.append(tuple(converted))
                raw_by_converted[tuple(converted)].append(annotation)
        assert Counter(reproduced) == Counter(tuple(b) for b in record['boxes']), record['image_path']
        assert hashlib.sha256(Path(record['image_path']).read_bytes()).hexdigest() == record['sha256']
        duplicates = []
        for box, entries in raw_by_converted.items():
            if len(entries)>1:
                distinct_raw = {tuple(a['bbox']) for a in entries}
                duplicates.append({'box_center_xywh':list(box), 'copies':len(entries),
                    'source_annotation_ids':[a['id'] for a in entries],
                    'raw_boxes_identical':len(distinct_raw)==1,
                    'source_boxes_xywh':[list(b) for b in sorted(distinct_raw)]})
        findings.append({'sha256':record['sha256'],'image':record['image_path'],
            'source_json':source,'source_json_sha256':source_hash,
            'source_image_id':record['source_image_id'],'split_group':record['split_group'],
            'original_count':len(record['boxes']),'unique_box_count':len(raw_by_converted),
            'duplicate_excess':len(record['boxes'])-len(raw_by_converted),
            'all_duplicates_identical_in_raw_source':all(d['raw_boxes_identical'] for d in duplicates),
            'duplicates':duplicates})
    del payload
findings.sort(key=lambda row:row['duplicate_excess'],reverse=True)
mapping={r['sha256']:r for r in affected}
for rank, row in enumerate(findings[:3]):
    r = mapping[row['sha256']]
    image=cv2.imread(r['image_path'])
    h,w=image.shape[:2]
    scale=min(900/w,800/h)
    frame=cv2.resize(image,(round(w*scale),round(h*scale)))
    for box,n in Counter(tuple(b) for b in r['boxes']).items():
        x,y,bw,bh=box
        color=(0,165,255) if n>1 else (0,200,0)
        cv2.rectangle(frame,(round((x-bw/2)*scale),round((y-bh/2)*scale)),(round((x+bw/2)*scale),round((y+bh/2)*scale)),color,1)
        if n>1:cv2.putText(frame,f'x{n}',(round(x*scale),round(y*scale)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
    canvas=np.full((frame.shape[0]+60,frame.shape[1],3),245,dtype=np.uint8)
    canvas[60:]=frame
    cv2.putText(canvas,f"{row['original_count']} labels / {row['unique_box_count']} distinct boxes",(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2)
    cv2.imwrite(str(out/f'{rank:02}.jpg'),canvas)

assert before_hash == hashlib.sha256(manifest.read_bytes()).hexdigest()
report={'created_at':datetime.now(timezone.utc).isoformat(),'split':'train','manifest_sha256':before_hash,
    'training_images':len(records),'training_annotations':sum(len(r['boxes']) for r in records),
    'affected_images':len(findings),'exact_duplicate_excess':sum(r['duplicate_excess'] for r in findings),
    'all_source_annotations_reproduced':True,'all_affected_image_hashes_verified':True,
    'all_duplicates_identical_in_raw_source':all(r['all_duplicates_identical_in_raw_source'] for r in findings),
    'raw_data_changed':False,'frozen_manifests_changed':False,'reserved_test_accessed':False,
    'application_status':'Audit only. No training or validation labels modified.',
    'findings':findings}
(out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='findings'},indent=2))
