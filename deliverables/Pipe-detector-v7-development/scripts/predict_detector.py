import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
"""Single-photo float32 inference for experimental pipe box-detector checkpoints."""
import argparse
import csv
from pathlib import Path
import cv2
import torch
from pipe_counter.detection import load_detector
from pipe_counter.utils import resolve_device,save_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(4)
    device=resolve_device(args.device)
    # Match the audited RTX 5090 validation runtime explicitly. CPU counts can
    # differ near thresholds; see reload_verification.json in this package.
    if device.type == 'cuda':
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
    model,checkpoint=load_detector(args.checkpoint,device)
    threshold=float(checkpoint['threshold'])
    source=Path(args.input)
    extensions={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
    images=[source] if source.is_file() else sorted(p for p in source.rglob('*') if p.suffix.lower() in extensions)
    if not images:raise ValueError('No images found')
    destination=Path(args.output_dir);destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    with torch.inference_mode():
        for i,path in enumerate(images):
            bgr=cv2.imread(str(path))
            if bgr is None:raise ValueError(f'Unreadable image: {path}')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            image=torch.from_numpy(rgb.copy()).permute(2,0,1).float().to(device)/255.
            result=model([image])[0]
            selected=result['scores']>=threshold
            count=int(selected.sum())
            payload={'image':str(path.resolve()),'detected_count':count,'threshold':threshold,
                'inference_precision':'float32','boxes':result['boxes'][selected].cpu().tolist(),
                'reference_cudnn_allow_tf32':True if device.type=='cuda' else None,
                'reference_cuda_matmul_allow_tf32':False if device.type=='cuda' else None,
                'scores':result['scores'][selected].cpu().tolist(),
                'at_detection_limit':len(result['scores'])>=checkpoint['config']['max_detections'],
                'production_approved':False}
            save_json(destination/f'{i:04d}_{path.stem}.json',payload)
            rows.append({'image':str(path.resolve()),'detected_count':count,'at_detection_limit':payload['at_detection_limit']})
            print(f'{path.name}: {count}',flush=True)
    with (destination/'predictions.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
