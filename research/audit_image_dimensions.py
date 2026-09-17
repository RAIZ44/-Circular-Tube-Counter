import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
from pipe_counter.utils import read_jsonl

project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
records=read_jsonl(project/'data/processed_v2/train.jsonl')+read_jsonl(project/'data/processed_v2/val.jsonl')
cv2.setNumThreads(1)
def inspect(record):
    image=cv2.imread(record['image_path'])
    if image is None:return {'image':record['image_path'],'error':'unreadable','split':record['split']}
    actual=[image.shape[1],image.shape[0]]
    expected=[record['width'],record['height']]
    if actual==expected:return None
    raw=cv2.imread(record['image_path'],cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
    return {'image':record['image_path'],'expected':expected,'decoded':actual,
        'ignore_orientation':[raw.shape[1],raw.shape[0]],'source_json':record['source_json'],
        'sha256':record['sha256'],'split':record['split'],'labels':len(record['boxes'])}
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=8) as pool:
        mismatches=[r for r in pool.map(inspect,records) if r]
    report={'images_checked':len(records),'test_accessed':False,'mismatches':mismatches}
    path=project/'runs/supervision/image_dimensions_audit.json'
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
