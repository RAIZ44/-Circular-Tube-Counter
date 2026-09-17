import pytest
import torch
from pipe_counter.inference import checkpoint_precision, probabilities, infer_heatmaps
from pipe_counter.peaks import peak_confidences, extract_peaks


def test_half_sigmoid_plateau_does_not_duplicate_peak_in_new_modes():
    logits = torch.tensor([[[[6., 6.00390625]]]], dtype=torch.float16)
    legacy = probabilities(logits, 'legacy_amp')
    corrected = probabilities(logits, 'amp')
    assert legacy[0, 0, 0, 0] == legacy[0, 0, 0, 1]
    assert corrected[0, 0, 0, 0] < corrected[0, 0, 0, 1]
    assert len(peak_confidences(legacy)[0]) == 2
    assert len(peak_confidences(corrected)[0]) == 1
    assert len(extract_peaks(corrected[0, 0], .9)) == 1


def test_precision_metadata_and_offline_cpu_inference():
    assert checkpoint_precision({}) == 'legacy_amp'
    assert checkpoint_precision({'config': {'inference_precision': 'float32'}}) == 'float32'
    with pytest.raises(ValueError):
        checkpoint_precision({'config': {'inference_precision': 'unknown'}})
    model = torch.nn.Conv2d(3, 1, 1).eval()
    inputs = torch.randn(1, 3, 8, 8)
    with torch.inference_mode():
        actual = infer_heatmaps(model, inputs, 'float32')
        assert torch.equal(actual, model(inputs).sigmoid())
    with pytest.raises(ValueError):
        infer_heatmaps(model, inputs, 'unknown')
