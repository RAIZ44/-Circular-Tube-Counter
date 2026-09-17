import json
from pathlib import Path
import cv2
import numpy as np
from pipe_counter.utils import read_jsonl

project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
out=Path(__file__).resolve().parent/'dimension_review';out.mkdir(exist_ok=True)
audit=json.loads((project/'runs/supervision/image_dimensions_audit.json').read_text())['mismatches']
records={r['sha256']:r for split in ('train','val') for r in read_jsonl(project/f'data/processed_v2/{split}.jsonl')}
panels=[]
for i,entry in enumerate(audit):
    r=records[entry['sha256']];im=cv2.imread(r['image_path']);h,w=im.shape[:2]
    row=[]
    for name in ('direct','cw','ccw','resize'):
        frame=cv2.resize(im,(480,round(480*h/w)))
        for x,y,bw,bh in r['boxes']:
            if name=='cw':x,y,bw,bh=r['height']-y,x,bh,bw
            elif name=='ccw':x,y,bw,bh=y,r['width']-x,bh,bw
            elif name=='resize':x,y,bw,bh=x*w/r['width'],y*h/r['height'],bw*w/r['width'],bh*h/r['height']
            s=480/w
            cv2.rectangle(frame,(round((x-bw/2)*s),round((y-bh/2)*s)),(round((x+bw/2)*s),round((y+bh/2)*s)),(0,255,0),1)
            cv2.circle(frame,(round(x*s),round(y*s)),2,(0,0,255),-1)
        title=np.full((48,480,3),245,np.uint8)
        cv2.putText(title,f'{i:02d} {name} labels={len(r["boxes"])} {r["split"]}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(10,10,10),1)
        row.append(np.vstack([title,frame]))
    path=out/f'{i:02d}.jpg';cv2.imwrite(str(path),np.hstack(row))
    panels.append({'index':i,'sha256':r['sha256'],'file':str(path),'labels':len(r['boxes']),'image':r['image_path'],'split':r['split']})
(out/'index.json').write_text(json.dumps(panels,indent=2))
print(json.dumps(panels))
