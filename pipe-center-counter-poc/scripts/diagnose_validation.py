"""Validation-only resolution, threshold, density and target-representation audit."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipe_counter.data import PipeCenterDataset
from pipe_counter.model import load_checkpoint_model
from pipe_counter.inference import checkpoint_precision, infer_heatmaps, PRECISIONS
from pipe_counter.peaks import count_metrics, peak_confidences
from pipe_counter.utils import resolve_device, save_json


def choose_metrics(values, counts):
    candidates = [count_metrics(values, counts, float(t)) for t in np.arange(.10, .701, .005)]
    mae = min(candidates, key=lambda m: (m['mean_absolute_error'], -m['exact_count_accuracy']))
    exact = min(candidates, key=lambda m: (-m['exact_count_accuracy'], m['mean_absolute_error']))
    return mae, exact


def summarize(rows):
    errors = np.asarray([row['predicted_count'] - row['true_count'] for row in rows])
    return {'images': len(rows), 'exact_count_accuracy': float(np.mean(errors == 0)),
            'within_one_accuracy': float(np.mean(np.abs(errors) <= 1)),
            'mean_absolute_error': float(np.abs(errors).mean()),
            'mean_signed_error': float(errors.mean()),
            'absolute_error_p95': float(np.percentile(np.abs(errors), 95)),
            'absolute_error_max': int(np.abs(errors).max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', default='data/processed_v2/val.jsonl')
    parser.add_argument('--image-sizes', nargs='+', type=int, default=[512, 768])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--precision', choices=PRECISIONS, default=None)
    args = parser.parse_args()
    if Path(args.manifest).name != 'val.jsonl':
        raise ValueError('This tuning script is restricted to val.jsonl; do not tune on the test set')
    device = resolve_device('cuda')
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    precision = args.precision or checkpoint_precision(checkpoint)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_results = []
    for size in args.image_sizes:
        if size % 16:
            raise ValueError('Image size must be divisible by 16')
        result_dir = output / str(size)
        result_dir.mkdir(exist_ok=True)
        if (result_dir / 'report.json').exists():
            raise FileExistsError('Preserve existing diagnostics; choose a new output directory')
        dataset = PipeCenterDataset(args.manifest, size, stride=model.output_stride)
        if any(r.get('split') != 'val' for r in dataset.records):
            raise ValueError('Manifest contains non-validation records')
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            shuffle=False, pin_memory=True)
        values, truth, oracle, unmasked = [], [], [], []
        started = time.perf_counter()
        with torch.inference_mode():
            for step, batch in enumerate(loader):
                inputs = batch['image'].to(device, non_blocking=True)
                heatmaps = infer_heatmaps(model, inputs, precision)
                values.extend(peak_confidences(heatmaps, bounds=batch['content_bounds']))
                unmasked.extend(peak_confidences(heatmaps, max_peaks=heatmaps.shape[-1]*heatmaps.shape[-2]))
                truth.extend(batch['count'].tolist())
                oracle_values = peak_confidences(batch['heatmap'], bounds=batch['content_bounds'])
                oracle.extend([int(np.count_nonzero(v >= .999)) for v in oracle_values])
                if step % 25 == 0:
                    print(f'{size}px: {min((step+1)*args.batch_size,len(dataset))}/{len(dataset)}', flush=True)
        elapsed = time.perf_counter() - started
        mae, exact = choose_metrics(values, truth)
        threshold = exact['threshold']
        rows = []
        for index, (record, scores, count, target_count) in enumerate(zip(dataset.records, values, truth, oracle)):
            predicted = int(np.count_nonzero(scores >= threshold))
            row = {'index': index, 'image': record['image_path'], 'source': Path(record['source_json']).parent.parent.name,
                   'source_file_name': record['source_file_name'], 'true_count': count,
                   'predicted_count': predicted, 'error': predicted-count,
                   'oracle_heatmap_count': target_count, 'oracle_error': target_count-count,
                   'annotation_conflict': bool(record.get('annotation_conflict')), 'split_group':record['split_group']}
            rows.append(row)
        with (result_dir/'per_image.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        np.savez_compressed(result_dir/'peak_scores.npz', **{f'image_{i}': v for i,v in enumerate(values)})
        densities={}
        for name,lower,upper in [('empty',0,0),('1-10',1,10),('11-50',11,50),('51-100',51,100),
                                 ('101-250',101,250),('251-500',251,500),('501+',501,100000)]:
            selected=[r for r in rows if lower<=r['true_count']<=upper]
            if selected: densities[name]=summarize(selected)
        sources={source:summarize([r for r in rows if r['source']==source]) for source in sorted({r['source'] for r in rows})}
        original_threshold=float(checkpoint['threshold'])
        corrected=count_metrics(values,truth,original_threshold)
        legacy=count_metrics(unmasked,truth,original_threshold)
        oracle_errors=np.asarray(oracle)-np.asarray(truth)
        result={'checkpoint':str(Path(args.checkpoint).resolve()), 'checkpoint_epoch':checkpoint['epoch'],
                'manifest':str(Path(args.manifest).resolve()), 'image_size':size,
                'evaluation_set':'validation_only', 'test_accessed':False,
                'inference_precision':precision,
                'inference_batch_size':args.batch_size,
                'original_threshold_with_consistent_postprocessing':corrected,
                'legacy_unmasked_original_threshold':legacy,
                'optimized_for_mae':mae,'optimized_for_exact_count':exact,
                'target_heatmap_recovery':{'exact_images':int((oracle_errors==0).sum()),
                    'images_with_count_mismatch':int((oracle_errors!=0).sum()),
                    'mean_absolute_error':float(np.abs(oracle_errors).mean()),
                    'note':'Count recovered from ideal training targets at 0.999; diagnoses quantization/collision/edge losses, not a model score or universal ceiling.'},
                'by_density':densities,'by_source':sources,
                'seconds_including_loading_and_diagnostics':elapsed,
                'worst_20':sorted(rows,key=lambda r:abs(r['error']),reverse=True)[:20]}
        save_json(result_dir/'report.json',result)
        all_results.append(result)
        print(json.dumps({k:result[k] for k in ('image_size','optimized_for_mae','optimized_for_exact_count','target_heatmap_recovery')},indent=2),flush=True)
    save_json(output/'summary.json',all_results)


if __name__=='__main__':
    main()
