import numpy as np
import pytest
import torch

from pipe_counter.data import content_bounds, letterbox
from pipe_counter.peaks import count_metrics, extract_peaks, peak_confidences
from pipe_counter.peaks import tune_threshold


@pytest.mark.parametrize("shape", [(21, 64), (64, 21), (64, 64)])
@pytest.mark.parametrize("threshold", [.3, .75])
def test_validation_and_prediction_agree_on_padding_and_peak_limit(shape, threshold):
    _, _, meta = letterbox(np.zeros((*shape, 3), np.uint8), [], 64)
    bounds = content_bounds(meta, 2)
    heatmap = torch.zeros(32, 32)
    heatmap[0, 0] = .99  # A padded location for non-square inputs.
    heatmap[15, 15] = .8
    heatmap[18, 18] = .6
    heatmap[31, 31] = .95
    peaks = extract_peaks(heatmap, threshold, max_peaks=1, bounds=bounds)
    values = peak_confidences(heatmap[None, None], bounds=torch.tensor([bounds]), max_peaks=1)
    metrics = count_metrics(values, [len(peaks)], threshold)
    assert metrics["exact_count_accuracy"] == 1
    assert len(peaks) == 1
    for x, y, _ in peaks:
        assert 0 <= (x * 2 - meta["pad_x"]) / meta["scale"] < shape[1]
        assert 0 <= (y * 2 - meta["pad_y"]) / meta["scale"] < shape[0]


def test_exact_objective_does_not_silently_optimize_mae():
    # Lower threshold wins two exact small scenes but overcounts a dense scene.
    values = [np.array([.2]), np.array([.2]), np.array([.2] * 100)]
    counts = [1, 1, 10]
    exact = tune_threshold(values, counts, [.1, .3], objective='exact_count_accuracy')
    assert exact['threshold'] == .1
    assert exact['exact_count_accuracy'] == pytest.approx(2/3)
    assert tune_threshold(values, counts, [.1, .3])['threshold'] == .3
    with pytest.raises(ValueError):
        tune_threshold(values, counts, [.1], objective='unknown')
