"""Validation-only experiment: overlapping crop context with disjoint counting regions."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from pipe_counter.data import letterbox, normalize_image
from pipe_counter.model import load_checkpoint_model
from pipe_counter.inference import checkpoint_precision, infer_heatmaps
from pipe_counter.peaks import peak_confidences
from pipe_counter.utils import read_jsonl, save_json
from diagnose_validation import choose_metrics, summarize


def tile_windows(width, height, grid=2, context=.2):
    cols = max(grid, math.ceil(width / height) * grid)
    rows = max(grid, math.ceil(height / width) * grid)
    for row in range(rows):
        for col in range(cols):
            x0, x1 = round(col * width / cols), round((col+1) * width / cols)
            y0, y1 = round(row * height / rows), round((row+1) * height / rows)
            if x1 <= x0 or y1 <= y0:
                continue
            dx, dy = math.ceil((x1-x0)*context), math.ceil((y1-y0)*context)
            yield (x0,y0,x1,y1), (max(0,x0-dx),max(0,y0-dy),min(width,x1+dx),min(height,y1+dy))


@torch.inference_mode()
def tile_scores(image, model, size, device, grid, precision='float32'):
    inputs, bounds = [], []
    for core, crop in tile_windows(image.shape[1],image.shape[0],grid):
        cx,cy,cr,cb=crop
        prepared,_,meta=letterbox(image[cy:cb,cx:cr],[],size)
        x0,y0,x1,y1=core
        s=model.output_stride
        bounds.append([(meta['pad_x']+(x0-cx)*meta['scale'])/s,
                       (meta['pad_y']+(y0-cy)*meta['scale'])/s,
                       (meta['pad_x']+(x1-cx)*meta['scale'])/s,
                       (meta['pad_y']+(y1-cy)*meta['scale'])/s])
        inputs.append(normalize_image(prepared))
    scores=[]
    for start in range(0,len(inputs),16):
        batch=torch.stack(inputs[start:start+16]).to(device)
        heatmaps=infer_heatmaps(model,batch,precision)
        scores.extend(peak_confidences(heatmaps,bounds=torch.tensor(bounds[start:start+16])))
    values=np.concatenate(scores)
    if len(values)>2000:
        values=np.sort(values)[-2000:]
    return values


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--grid',type=int,default=2)
    args=parser.parse_args()
    if args.grid<1: raise ValueError('Grid must be positive')
    records=read_jsonl('data/processed_v2/val.jsonl')
    assert all(r['split']=='val' for r in records)
    device=torch.device('cuda')
    model,checkpoint=load_checkpoint_model(args.checkpoint,device)
    precision=checkpoint_precision(checkpoint)
    size=int(checkpoint['config']['image_size'])
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'report.json').exists(): raise FileExistsError('Use a new output directory')
    values=[]
    for i,r in enumerate(records):
        image=cv2.imread(r['image_path'])
        if image is None: raise RuntimeError(r['image_path'])
        values.append(tile_scores(image,model,size,device,args.grid,precision))
        if i%100==0: print(f'tiled {i+1}/{len(records)}',flush=True)
    truth=[len(r['boxes']) for r in records]
    mae,exact=choose_metrics(values,truth)
    rows=[]
    for r,v,t in zip(records,values,truth):
        predicted=int((v>=exact['threshold']).sum())
        rows.append({'image':r['image_path'],'true_count':t,'predicted_count':predicted,'error':predicted-t})
    with (output/'per_image.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'peak_scores.npz',**{f'image_{i}':v for i,v in enumerate(values)})
    report={'evaluation_set':'validation_only','test_accessed':False,'grid':args.grid,
            'inference_precision':precision,
            'checkpoint':str(Path(args.checkpoint).resolve()),'optimized_for_mae':mae,
            'optimized_for_exact_count':exact,'errors':summarize(rows),
            'note':'Context overlaps; each detection is assigned to one half-open core region. Experimental crop mode, not production inference.'}
    save_json(output/'report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
