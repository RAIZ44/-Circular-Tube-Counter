import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from pipe_counter.data import letterbox, normalize_image, content_bounds
from pipe_counter.inference import checkpoint_precision, infer_heatmaps
from pipe_counter.model import load_checkpoint_model
from pipe_counter.peaks import extract_peaks

def main():
    package = Path(__file__).resolve().parents[1]/'outputs/Pipe-counter-development-candidate'
    examples = json.loads((package/'regression_examples.json').read_text())
    decoded = [cv2.imread(str(package/r['packaged_image'])) for r in examples]
    result = {'scope':'12 supplied validation regression fixtures; not independent accuracy or an Azure benchmark', 'devices':{}}
    for device_name in ('cuda','cpu'):
        device = torch.device(device_name)
        model, checkpoint = load_checkpoint_model(package/'candidate.pt', device)
        precision = checkpoint_precision(checkpoint)
        def predict(image):
            prepared, _, meta = letterbox(image, [], 512)
            tensor = normalize_image(prepared)[None].to(device)
            heatmap = infer_heatmaps(model, tensor, precision)[0,0].cpu()
            return len(extract_peaks(heatmap, checkpoint['threshold'], bounds=content_bounds(meta,2)))
        with torch.inference_mode():
            for _ in range(3): predict(decoded[0])
            timings, counts = [], []
            for image in decoded:
                start = time.perf_counter()
                counts.append(predict(image))
                timings.append((time.perf_counter()-start)*1000)
        result['devices'][device_name] = {
            'counts':counts, 'matches_expected':sum(c == int(r['predicted_count']) for c,r in zip(counts,examples)),
            'p50_ms':float(np.median(timings)), 'p95_ms':float(np.percentile(timings,95)),
            'timing_scope':'warm serial preprocessing, model, CPU peak extraction; excludes disk decode, model loading and rendering'}
        del model
    result['cuda_cpu_count_matches'] = sum(a == b for a,b in zip(result['devices']['cuda']['counts'],result['devices']['cpu']['counts']))
    (package/'runtime_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
