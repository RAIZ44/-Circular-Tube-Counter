import csv
import json
from pathlib import Path
import numpy as np
from pipe_counter.peaks import tune_threshold,count_metrics
from pipe_counter.utils import read_jsonl

project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
records=read_jsonl(project/'data/processed_v2/val.jsonl')
baseline=list(csv.DictReader((project/'runs/validation_diagnostics/v4_best_float32_single/512/per_image.csv').open()))
detector=list(csv.DictReader((project/'runs/pipe_detector_v7_finetune/best_validation_per_image.csv').open()))
assert [r['image'] for r in detector]==[r['image'] for r in baseline]==[r['image_path'] for r in records]
scores=np.load(project/'runs/pipe_detector_v7_finetune/best_validation_scores.npz')
values=[scores[f'image_{i}'] for i in range(len(records))]
truth=np.array([len(r['boxes']) for r in records])
a=np.array([int(r['predicted_count']) for r in baseline]);b=np.array([int(r['predicted_count']) for r in detector])
def summarize(pred,idx):
    e=pred[idx]-truth[idx]
    return {'images':int(idx.sum()),'exact':float((e==0).mean()),'within_one':float((np.abs(e)<=1).mean()),'mae':float(np.abs(e).mean()),'bias':float(e.mean()),'p95':float(np.percentile(np.abs(e),95))}
result={'evaluation':'validation_only','test_accessed':False,
    'detector_mae_optimized':tune_threshold(values,truth,np.arange(.01,.991,.005)),
    'both_exact':int(((a==truth)&(b==truth)).sum()),'baseline_only_exact':int(((a==truth)&(b!=truth)).sum()),
    'detector_only_exact':int(((b==truth)&(a!=truth)).sum()),'by_density':{},'by_source':{}}
for label,lo,hi in [('0-10',0,10),('11-50',11,50),('51-100',51,100),('101-250',101,250),('251-500',251,500),('501+',501,10000)]:
    idx=(truth>=lo)&(truth<=hi)
    result['by_density'][label]={'baseline':summarize(a,idx),'detector':summarize(b,idx)}
sources=np.array([Path(r['source_json']).parent.parent.name for r in records])
for source in sorted(set(sources)):
    idx=sources==source;result['by_source'][source]={'baseline':summarize(a,idx),'detector':summarize(b,idx)}
result['worst_detector']=[{'index':int(i),'image':records[i]['image_path'],'true':int(truth[i]),'detector':int(b[i]),'baseline':int(a[i])} for i in np.argsort(np.abs(b-truth))[-10:][::-1]]
path=project/'runs/supervision/detector_v7_comparison.json';path.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

