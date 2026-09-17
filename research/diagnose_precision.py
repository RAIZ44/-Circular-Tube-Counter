"""Compare numerical precision on validation only; never modify a checkpoint."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from pipe_counter.data import PipeCenterDataset
from pipe_counter.model import load_checkpoint_model
from pipe_counter.peaks import peak_confidences, count_metrics, tune_threshold

def main():
    project = Path(r'C:\Users\Owner\Projects\Pipe Counter New\pipe-center-counter-poc')
    output = project / 'runs/validation_diagnostics/v4_precision'
    output.mkdir(exist_ok=False)
    dataset = PipeCenterDataset(project / 'data/processed_v2/val.jsonl', 512, stride=2)
    assert all(r['split'] == 'val' for r in dataset.records)
    model, checkpoint = load_checkpoint_model(project / 'runs/pipe_center_v4_resnet18_exact/best.pt', torch.device('cuda'))
    methods = {name: [] for name in ('amp_half_sigmoid', 'amp_float_sigmoid', 'full_float')}
    truth = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True):
            images = batch['image'].cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(images)
            maps = {'amp_half_sigmoid': logits.sigmoid(), 'amp_float_sigmoid': logits.float().sigmoid(),
                    'full_float': model(images).sigmoid()}
            for name, heatmaps in maps.items():
                methods[name].extend(peak_confidences(heatmaps, bounds=batch['content_bounds']))
            truth.extend(batch['count'].tolist())
    threshold = float(checkpoint['threshold'])
    original_counts = np.array([np.count_nonzero(v >= threshold) for v in methods['amp_half_sigmoid']])
    results = {'evaluation_set': 'validation_only', 'test_accessed': False, 'checkpoint_threshold': threshold, 'methods': {}}
    for name, values in methods.items():
        counts = np.array([np.count_nonzero(v >= threshold) for v in values])
        results['methods'][name] = {
            'at_frozen_threshold': count_metrics(values, truth, threshold),
            'optimized_exact': tune_threshold(values, truth, np.arange(.1, .701, .005), objective='exact_count_accuracy'),
            'images_changed_vs_amp': int(np.count_nonzero(counts != original_counts)),
            'max_count_change_vs_amp': int(np.max(np.abs(counts-original_counts)))}
    (output / 'report.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
