import csv,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

p=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
w=Path(__file__).resolve().parents[1]
s=p/'runs/supervision'
now=datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,obj): path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
v8=p/'runs/pipe_detector_v8_960'
metrics=read(v8/'best_validation_metrics.json')
history=list(csv.DictReader((v8/'history.csv').open()))
best=max(history,key=lambda r:(float(r['exact_count_accuracy']),-float(r['mean_absolute_error'])))
rows={name:list(csv.DictReader((p/'runs'/run/'best_validation_per_image.csv').open())) for name,run in [('v7','pipe_detector_v7_finetune'),('v8','pipe_detector_v8_960')]}
assert [r['image'] for r in rows['v7']]==[r['image'] for r in rows['v8']]
truth=np.array([int(r['true_count']) for r in rows['v8']])
assert len(truth)==1332
comparison={'evaluation':'validation_only','test_evaluated':False,'best_epoch':int(best['epoch']),'metrics':metrics,'by_density':{}}
for band,lo,hi in [('0-10',0,10),('11-50',11,50),('51-100',51,100),('101-250',101,250),('251-500',251,500),('501+',501,10000)]:
    selected=(truth>=lo)&(truth<=hi)
    comparison['by_density'][band]={}
    for name,data in rows.items():
        error=np.array([int(r['predicted_count']) for r in data])-truth
        comparison['by_density'][band][name]={'images':int(selected.sum()),'exact':float((error[selected]==0).mean()),'mae':float(np.abs(error[selected]).mean())}
write(s/'detector_v8_comparison.json',comparison)
experiments=read(s/'experiments.json')
record=experiments.pop('next_experiment')
assert record['run']=='pipe_detector_v8_960'
record.update(experiment=record['run'],status='completed',epochs_completed=len(history),best_epoch=int(best['epoch']),metrics=metrics,last_epoch_metrics={k:float(v) for k,v in history[-1].items()},decision='No promotion: best exact accuracy and MAE both worse than v7. Last epoch lowers MAE to 7.9444 but exact accuracy remains lower. Preserve best and last; no extension.')
experiments['results'].append(record)
queue=read(s/'architecture_queue.json')
smoke=read(w/'work/fcos_gpu_smoke.json')
assert smoke['single_photo_float32_inference_passed'] and all(b['optimizer_step_passed'] for b in smoke['batches'])
queue['next']['preparation_checks']['gpu_dense_smoke_passed']=True
queue['next']['preparation_checks']['gpu_dense_smoke']=smoke
queue['next']['status']='ready_to_launch'
queue['updated_at']=now
queue['policy']='One GPU job at a time; v8 completed normally. Preserve existing runs and verify no live owner before archiving its lock. FCOS GPU preflight passed.'
write(s/'architecture_queue.json',queue)
experiments.update(updated_at=now,next_experiment=queue['next'],next_action='Launch and supervise bounded FCOS comparison; reserved test remains unused.')
write(s/'experiments.json',experiments)
state=read(s/'status.json')
state.pop('provisional_exact_count_target',None)
state.update(minimum_exact_count_accuracy=.95,acceptance_criteria_file=str(p/'ACCEPTANCE_CRITERIA.json'),operating_scope=read(p/'OPERATING_SCOPE.json'),active_training=False)
write(s/'status.json',state)
note=f'''\n\n## V8 complete; FCOS GPU preflight passed ({now})\n\nV8 stopped normally after four epochs, best epoch two: 394/1332 exactly\ncorrect (29.5796%), MAE 9.6321. V7 remains 397/1332 (29.8048%), MAE\n8.3529; v6 calibrated remains best exact at 398/1332 (29.8799%). V8 last\nepoch had MAE 7.9444 and 392 exact photos; preserve both checkpoints without\npromotion or extension. Density comparison: runs/supervision/detector_v8_comparison.json.\n\nFCOS 960px CUDA smoke passed dense batches with 1523 labels each and a\n10-label/empty batch, including optimizer steps and float32 inference. Peak\nGPU allocation 2.13 GiB. This establishes execution feasibility, not accuracy.\nProceed with the prepared fresh-pretrained FCOS run: batch two, LR 1e-4,\nmaximum 12 epochs, patience four, NMS .3, unchanged training/validation labels.\n95% exact-count target is authoritative; test remains reserved.\n'''
with (p/'PRODUCTION_READINESS.md').open('a',encoding='utf-8') as f:f.write(note)
with (w/'outputs/Validation-progress.md').open('a',encoding='utf-8') as f:f.write(note)
architecture=w/'outputs/Architecture-comparison.md'
text=architecture.read_text(encoding='utf-8').replace('Training now, at most six epochs','Completed; 29.58% exact, no promotion').replace('Implemented and CPU-tested; queued after the active run','CPU and dense GPU checks passed; launching bounded training')
text=text.replace('Full-resolution GPU preflight remains pending until the\ncurrent run finishes. The planned comparison is capped at 12 epochs with early\nstopping; it has not started or demonstrated an accuracy improvement yet.','Full-resolution GPU preflight passed on the RTX 5090, including two images\nwith 1,523 labels each. The comparison is capped at 12 epochs with early\nstopping; it has not yet demonstrated an accuracy improvement.')
architecture.write_text(text,encoding='utf-8')
print(json.dumps(comparison,indent=2))
