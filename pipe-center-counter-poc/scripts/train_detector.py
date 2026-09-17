"""Bounded detector comparison; validation chooses the threshold, never test data."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from pipe_counter.detection import ARCHITECTURE, DETECTOR_ARCHITECTURES, DetectionDataset, create_detector, collate_detection, validate_detector_initialization, configure_detector, pipe_scores
from pipe_counter.peaks import tune_threshold
from pipe_counter.utils import resolve_device,set_seed,save_json
from pipe_counter.splitting import source_key


@torch.inference_mode()
def validate_detector(model,loader,device):
    model.eval()
    scores,truth=[],[]
    for images,targets in loader:
        # Full-precision single-photo inference, matching the planned runtime.
        outputs=model([image.to(device) for image in images])
        for output,target in zip(outputs,targets):
            scores.append(pipe_scores(output).cpu().numpy().astype(np.float32))
            truth.append(len(target['boxes']))
    return scores,truth


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest-dir',default='data/processed_v2')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--image-size',type=int,default=640)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--patience',type=int,default=3)
    parser.add_argument('--learning-rate',type=float,default=.0002)
    parser.add_argument('--num-workers',type=int,default=4)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--architecture',choices=DETECTOR_ARCHITECTURES,default=ARCHITECTURE)
    parser.add_argument('--pretrained',action='store_true')
    parser.add_argument('--init-checkpoint',default=None)
    parser.add_argument('--box-nms-threshold',type=float,default=.5)
    parser.add_argument('--geometry-overrides',default='data/geometry_overrides_v1.json')
    parser.add_argument('--selection-metric',choices=['exact_count_accuracy'],default='exact_count_accuracy')
    args=parser.parse_args()
    if args.init_checkpoint and args.pretrained:raise ValueError('Choose pretrained or init-checkpoint, not both')
    if not 0<args.box_nms_threshold<=1:raise ValueError('NMS IoU must be in (0,1]')
    if args.epochs<1 or args.batch_size<1: raise ValueError('Positive epochs/batch required')
    output=Path(args.output_dir)
    if (output/'history.csv').exists() or (output/'best.pt').exists():
        raise FileExistsError('Preserve existing runs')
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'train_config.json',vars(args))
    set_seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.hub.set_dir(Path(__file__).resolve().parents[1]/'.cache/torch')
    device=resolve_device(args.device)
    geometry_path=Path(args.geometry_overrides)
    geometry_file=geometry_path if geometry_path.exists() else None
    train=DetectionDataset(Path(args.manifest_dir)/'train.jsonl',augment=True,geometry_overrides=geometry_file)
    val=DetectionDataset(Path(args.manifest_dir)/'val.jsonl',geometry_overrides=geometry_file)
    if any(r.get('split')!='train' for r in train.records) or any(r.get('split')!='val' for r in val.records):
        raise ValueError('Only train/validation manifests may be used here')
    train_hashes={r['sha256'] for r in train.records}
    train_groups={r['split_group'] for r in train.records}
    if any(r['sha256'] in train_hashes or r['split_group'] in train_groups for r in val.records):
        raise ValueError('Train/validation overlap detected')
    records=train.records+val.records
    development={'sha256':sorted({r['sha256'] for r in records}),
        'source_keys':sorted({k for r in records for k in r.get('source_keys',[source_key(r['source_file_name'])])}),
        'perceptual_group':sorted({r['perceptual_group'] for r in records if r.get('perceptual_group')}),
        'note':'Training and validation both influence selection.'}
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,
        collate_fn=collate_detection,pin_memory=device.type=='cuda')
    val_loader=DataLoader(val,batch_size=1,shuffle=False,num_workers=args.num_workers,
        collate_fn=collate_detection,pin_memory=device.type=='cuda')
    model=create_detector(args.image_size,args.pretrained,architecture=args.architecture).to(device)
    configure_detector(model,args.box_nms_threshold)
    initial_metadata=None
    if args.init_checkpoint:
        parent=torch.load(args.init_checkpoint,map_location='cpu',weights_only=False)
        geometry_hash=hashlib.sha256(geometry_file.read_bytes()).hexdigest() if geometry_file else None
        validate_detector_initialization(parent,development,geometry_hash,architecture=args.architecture)
        model.load_state_dict(parent['model'])
        initial_metadata={'path':str(Path(args.init_checkpoint).resolve()),
            'sha256':hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest(),
            'epoch':parent['epoch'],'pretrained_detector':parent['config'].get('pretrained_detector',False),
            'note':'Weights initialized from parent; optimizer, scheduler, and scaler reset for a new bounded fine-tune.'}
        del parent
    backbone=[p for n,p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]
    heads=[p for n,p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':args.learning_rate*.1},
        {'params':heads,'lr':args.learning_rate}],weight_decay=.0001,foreach=False)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=1,factor=.5)
    scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda')
    best_key=None;stale=0
    config={'architecture':args.architecture,'image_size':args.image_size,'inference_precision':'float32',
        'inference_batch_size':1,'input_normalization':'RGB float [0,1]; model applies ImageNet normalization and resize',
        'anchor_sizes':[16,32,64,128,256],'box_nms_threshold':args.box_nms_threshold,'max_detections':2000,
        'minimum_score':.01,'selection_metric':'exact_count_accuracy','pretrained_detector':args.pretrained}
    config.update(cudnn_allow_tf32=True,cuda_matmul_allow_tf32=False,foreground_label=1)
    if args.architecture=='FCOSResNet50FPN':
        config['anchor_sizes']=[8,16,32,64,128]
        config['anchor_free']=True
        config['topk_candidates']=10000
    if initial_metadata:
        config['initialization']=initial_metadata
        config['pretrained_detector']=initial_metadata['pretrained_detector']
    if geometry_file:
        config['geometry_overrides_sha256']=hashlib.sha256(geometry_file.read_bytes()).hexdigest()
        config['geometry_overrides']=json.loads(geometry_file.read_text())
        save_json(output/'geometry_overrides.json',config['geometry_overrides'])
    print(f'Detector CUDA experiment: train={len(train)} val={len(val)} size={args.image_size}',flush=True)
    for epoch in range(1,args.epochs+1):
        model.train()
        # Fixed pretrained running statistics are more stable for small batches.
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.eval()
        loss_sum=0.;images_seen=0;started=time.monotonic()
        for step,(images,targets) in enumerate(train_loader,1):
            images=[image.to(device,non_blocking=True) for image in images]
            targets=[{k:v.to(device,non_blocking=True) for k,v in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                losses=model(images,targets)
                loss=sum(losses.values())
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite detector loss: {losses}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            scaler.step(optimizer);scaler.update()
            loss_sum+=float(loss.detach())*len(images);images_seen+=len(images)
            if step==1 or step%100==0:
                print(f'Epoch {epoch}/{args.epochs} batch {step}/{len(train_loader)} loss={loss_sum/images_seen:.5f} elapsed={time.monotonic()-started:.1f}s',flush=True)
        scores,truth=validate_detector(model,val_loader,device)
        lower_threshold=.01 if args.architecture=='FCOSResNet50FPN' else .05
        metrics=tune_threshold(scores,truth,np.arange(lower_threshold,.951,.005),objective='exact_count_accuracy')
        key=(metrics['exact_count_accuracy'],-metrics['mean_absolute_error'])
        improved=best_key is None or key>best_key
        if improved:best_key=key;stale=0
        else:stale+=1
        scheduler.step(metrics['exact_count_accuracy'])
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
            'threshold':metrics['threshold'],'metrics':metrics,'config':config,'development_data':development}
        for name in (['last.pt','best.pt'] if improved else ['last.pt']):
            temporary=output/(name+'.tmp');torch.save(checkpoint,temporary);temporary.replace(output/name)
        if improved:
            save_json(output/'best_validation_metrics.json',metrics)
            np.savez_compressed(output/'best_validation_scores.npz',**{f'image_{i}':s for i,s in enumerate(scores)})
            with (output/'best_validation_per_image.csv').open('w',newline='',encoding='utf-8') as handle:
                writer=csv.DictWriter(handle,fieldnames=['image','true_count','predicted_count','error','split_group'])
                writer.writeheader()
                for record,scores_i,count in zip(val.records,scores,truth):
                    predicted=int((scores_i>=metrics['threshold']).sum())
                    writer.writerow({'image':record['image_path'],'true_count':count,'predicted_count':predicted,
                        'error':predicted-count,'split_group':record['split_group']})
        row={'epoch':epoch,'train_loss':loss_sum/images_seen,**metrics,'seconds':time.monotonic()-started}
        with (output/'history.csv').open('a',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row))
            if epoch==1:writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row),flush=True)
        if stale>=args.patience:break


if __name__=='__main__':main()
