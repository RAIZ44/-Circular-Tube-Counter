import sys, json, time, hashlib, statistics
from pathlib import Path
from datetime import datetime, timezone
workspace=Path(__file__).resolve().parents[1]
package=workspace/'outputs/Pipe-detector-v7-development'
sys.path.insert(0,str(package/'src'))
import cv2, torch
import pipe_counter.detection as detection

assert Path(detection.__file__).resolve().is_relative_to(package)
torch.set_num_threads(4)
fixtures=json.loads((package/'regression_examples.json').read_text())
report={'created_at':datetime.now(timezone.utc).isoformat(),'bundled_source':str(Path(detection.__file__).resolve()),
    'fixture_count':len(fixtures),'test_evaluated':False,'production_ready':False,
    'environment':{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),'threads':torch.get_num_threads(),
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32},'cli_checks':{}}
for device in ['cuda','cpu']:
    checks=[]
    for i,f in enumerate(fixtures):
        image_path=package/f['file']
        assert hashlib.sha256(image_path.read_bytes()).hexdigest()==f['sha256']
        prediction=json.loads((package/f'verification_{device}'/f'{i:04}_{image_path.stem}.json').read_text())
        checks.append({'fixture':f['file'],'expected':f['expected_prediction'],
            'actual':prediction['detected_count'],'matches':prediction['detected_count']==f['expected_prediction']})
    report['cli_checks'][device]={'matching_counts':sum(c['matches'] for c in checks),'checks':checks}
assert report['cli_checks']['cuda']['matching_counts']==len(fixtures)
parent=Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc\runs\pipe_detector_v7_finetune\best.pt')
original=torch.load(parent,map_location='cpu',weights_only=False)
packaged=torch.load(package/'candidate.pt',map_location='cpu',weights_only=False)
assert original['config']==packaged['config'] and original['threshold']==packaged['threshold']
assert all(torch.equal(v,packaged['model'][k]) for k,v in original['model'].items())
report['all_weights_and_inference_settings_identical_to_parent']=True
del original,packaged

def tensor_for(path,device):
    bgr=cv2.imread(str(path));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).permute(2,0,1).float().to(device)/255.

model,checkpoint=detection.load_detector(package/'candidate.pt',torch.device('cuda'))
threshold=checkpoint['threshold']
timings=[]
with torch.inference_mode():
    warm=tensor_for(package/fixtures[0]['file'],'cuda')
    for _ in range(3):model([warm])
    for f in fixtures:
        image=tensor_for(package/f['file'],'cuda')
        torch.cuda.synchronize();start=time.perf_counter()
        pred=model([image])[0];torch.cuda.synchronize()
        timings.append((time.perf_counter()-start)*1000)
        assert int((pred['scores']>=threshold).sum())==f['expected_prediction']
    diagnostics=[]
    image=tensor_for(package/fixtures[11]['file'],'cuda')
    for strict in [False,True]:
        torch.backends.cudnn.allow_tf32=not strict
        torch.backends.cuda.matmul.allow_tf32=False
        for repeat in range(2):
            pred=model([image])[0]
            scores=pred['scores'].cpu().tolist()
            near=sorted(scores,key=lambda value:abs(value-threshold))[:5]
            diagnostics.append({'device':'cuda','cudnn_allow_tf32':not strict,'repeat':repeat,
                'count':sum(value>=threshold for value in scores),'scores_nearest_threshold':near})
report['cuda_warm_inference_ms']={'median':statistics.median(timings),'max':max(timings),
    'sample_count':len(timings),'per_image':timings,
    'scope':'One synchronized forward pass per fixture after three warmups; includes model normalization/resize, excludes decoding, tensor creation, startup and output writing. Local RTX 5090 only; not an Azure SLA.'}
report['cross_device_diagnostic']={'fixture':fixtures[11]['file'],
    'cpu_count':report['cli_checks']['cpu']['checks'][11]['actual'],
    'gpu_runs':diagnostics,
    'interpretation':'A CPU/GPU count difference exists near the confidence threshold. Do not claim cross-device count parity. Keep saved CUDA reference behavior and validate any target runtime before release.'}
(package/'reload_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['cli_checks','cuda_warm_inference_ms']},indent=2))
print(json.dumps({'cuda_matches':report['cli_checks']['cuda']['matching_counts'],
    'cpu_matches':report['cli_checks']['cpu']['matching_counts'],
    'gpu_median_ms':report['cuda_warm_inference_ms']['median']}))
