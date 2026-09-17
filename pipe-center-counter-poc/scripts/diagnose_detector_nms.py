"""Validation-only comparison of duplicate-box suppression, with one model pass."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms
from pipe_counter.detection import DetectionDataset,load_detector,collate_detection
from pipe_counter.peaks import tune_threshold


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    model,checkpoint=load_detector(args.checkpoint,torch.device('cuda'))
    dataset=DetectionDataset('data/processed_v2/val.jsonl',geometry_overrides='data/geometry_overrides_v1.json')
    assert all(r['split']=='val' for r in dataset.records)
    model.roi_heads.nms_thresh=1.0
    model.roi_heads.detections_per_img=10000
    variants={str(t):[] for t in (.2,.3,.4,.5,.6,.7)}
    truth=[]
    with torch.inference_mode():
        for i,(images,targets) in enumerate(DataLoader(dataset,batch_size=1,num_workers=4,collate_fn=collate_detection)):
            pred=model([images[0].cuda()])[0]
            for value,scores in variants.items():
                keep=nms(pred['boxes'],pred['scores'],float(value))[:2000]
                scores.append(pred['scores'][keep].cpu().numpy())
            truth.append(len(targets[0]['boxes']))
            if i%200==0:print(f'NMS comparison {i+1}/{len(dataset)}',flush=True)
    report={'evaluation_set':'validation_only','test_accessed':False,'checkpoint':str(Path(args.checkpoint).resolve()),'variants':{}}
    for value,scores in variants.items():
        metrics=tune_threshold(scores,truth,np.arange(.01,.991,.005),objective='exact_count_accuracy')
        report['variants'][value]=metrics
        np.savez_compressed(output/f'scores_iou_{value}.npz',**{f'image_{i}':v for i,v in enumerate(scores)})
        with (output/f'per_image_iou_{value}.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=['image','true_count','predicted_count','error'])
            writer.writeheader()
            for record,values,count in zip(dataset.records,scores,truth):
                predicted=int((values>=metrics['threshold']).sum())
                writer.writerow({'image':record['image_path'],'true_count':count,'predicted_count':predicted,'error':predicted-count})
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
