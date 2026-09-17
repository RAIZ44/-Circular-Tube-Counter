"""One bounded 960px detector comparison on frozen validation images only."""
import csv, hashlib, json, os, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from pipe_counter.detection import DetectionDataset, load_detector
from pipe_counter.peaks import count_metrics, tune_threshold

project=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
output=project/'runs/pipe_detector_v7_960_diagnostic'
status_path=project/'runs/supervision/status.json'
manifest=project/'data/processed_v2/val.jsonl'
parent=project/'runs/pipe_detector_v7_finetune/best.pt'
def now():return datetime.now(timezone.utc).isoformat()
def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)
def main():
    output.mkdir(exist_ok=False)
    state=json.loads(status_path.read_text())
    if state.get('active_training'):raise RuntimeError('Training already active')
    state.update(phase='validation_diagnostic',updated_at=now(),diagnostic_pid=os.getpid(),
        active_diagnostic=True,diagnostic_dir=str(output),
        next_action='Compare one 960px detector input against the 640px reference, including density error. Labels unchanged; counting-rule questions do not block independent work.')
    save(status_path,state)
    try:
        assert torch.cuda.is_available()
        torch.set_num_threads(4)
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cuda.matmul.allow_tf32=False
        data=DetectionDataset(manifest,geometry_overrides=project/'data/geometry_overrides_v1.json')
        assert len(data)==1332 and all(r['split']=='val' for r in data.records)
        model,checkpoint=load_detector(parent,torch.device('cuda'))
        assert checkpoint['config']['image_size']==640
        model.transform.min_size=(960,);model.transform.max_size=960
        provenance={'started_at':now(),'checkpoint_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
            'validation_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'image_size':960,'baseline_image_size':640,'weights_changed':False,'labels_changed':False,
            'test_evaluated':False,'cudnn_allow_tf32':True,'cuda_matmul_allow_tf32':False,
            'hypothesis':'More pixels may recover small distant/dense pipe ends missed by the 640px detector.',
            'decision_rule':'Consider bounded adaptation only if exact accuracy improves at least 2 percentage points, or 501+ MAE improves at least 20% with no overall-MAE regression. Reject unadapted larger input otherwise; no automatic promotion.'}
        save(output/'provenance.json',provenance)
        scores=[];truth=[];started=time.monotonic()
        with torch.inference_mode():
            for i in range(len(data)):
                image,target=data[i]
                pred=model([image.cuda()])[0]
                scores.append(pred['scores'].cpu().numpy().astype(np.float32));truth.append(len(target['boxes']))
                if (i+1)%100==0 or i==len(data)-1:
                    state.update(updated_at=now(),diagnostic_completed_images=i+1)
                    save(status_path,state)
                    print(f'960px validation {i+1}/{len(data)}; {time.monotonic()-started:.1f}s',flush=True)
        threshold=checkpoint['threshold']
        fixed=count_metrics(scores,truth,threshold)
        calibrated=tune_threshold(scores,truth,np.arange(.05,.951,.005),objective='exact_count_accuracy')
        archive=np.load(project/'runs/pipe_detector_v7_finetune/best_validation_scores.npz')
        reference=[archive[f'image_{i}'] for i in range(len(data))]
        baseline=count_metrics(reference,truth,threshold)
        assert baseline==checkpoint['metrics']
        slices={}
        for low,high in [(0,10),(11,50),(51,100),(101,250),(251,500),(501,10000)]:
            ids=[i for i,n in enumerate(truth) if low<=n<=high]
            slices[f'{low}-{high}']={
                'baseline':count_metrics([reference[i] for i in ids],[truth[i] for i in ids],threshold),
                'higher_resolution':count_metrics([scores[i] for i in ids],[truth[i] for i in ids],calibrated['threshold'])}
        dense=slices['501-10000']
        promising=(calibrated['exact_count_accuracy']>=baseline['exact_count_accuracy']+.02 or
            (dense['higher_resolution']['mean_absolute_error']<=.8*dense['baseline']['mean_absolute_error'] and
             calibrated['mean_absolute_error']<=baseline['mean_absolute_error']))
        report={'evaluation':'validation_only','test_evaluated':False,'baseline':baseline,
            'fixed_threshold_960':fixed,'calibrated_960':calibrated,'by_density':slices,
            'elapsed_seconds':time.monotonic()-started,'meets_predeclared_adaptation_rule':promising}
        np.savez_compressed(output/'scores.npz',**{f'image_{i}':s for i,s in enumerate(scores)})
        with (output/'per_image.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=['image','true_count','baseline_count','higher_resolution_count','split_group'])
            writer.writeheader()
            for r,t,a,b in zip(data.records,truth,reference,scores):
                writer.writerow({'image':r['image_path'],'true_count':t,'baseline_count':int((a>=threshold).sum()),
                    'higher_resolution_count':int((b>=calibrated['threshold']).sum()),'split_group':r['split_group']})
        save(output/'report.json',report)
        state.update(phase='validation_review',active_diagnostic=False,updated_at=now(),
            diagnostic_report=str(output/'report.json'),next_action='Review the completed 960px comparison against the predeclared decision rule before any further training.')
        save(status_path,state)
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as error:
        state.update(phase='diagnostic_failed',active_diagnostic=False,updated_at=now(),
            diagnostic_error=f'{type(error).__name__}: {error}')
        save(status_path,state)
        raise
if __name__=='__main__':main()
