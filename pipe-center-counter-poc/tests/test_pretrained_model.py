from unittest.mock import patch

import pytest
import torch

from pipe_counter.heatmap import modified_focal_loss
from pipe_counter.model import create_model, load_checkpoint_model


def test_resnet_center_model_backward_and_offline_reload(tmp_path):
    model = create_model('ResNet18CenterNet')
    x = torch.rand(2,3,96,128) * 2 - 1
    output = model(x)
    assert output.shape == (2,1,48,64)
    target = torch.zeros_like(output)
    target[:,:,24,32] = 1
    loss = modified_focal_loss(output, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.backbone.conv1.weight.grad is not None
    assert torch.isfinite(model.backbone.conv1.weight.grad).all()
    model.eval()
    with torch.inference_mode(): expected = model(x)
    checkpoint = tmp_path/'model.pt'
    torch.save({'model':model.state_dict(),'config':{'architecture':'ResNet18CenterNet'}}, checkpoint)
    # Loading a delivered model must not download external weights.
    with patch('torch.hub.download_url_to_file', side_effect=AssertionError('Unexpected network access')):
        restored, _ = load_checkpoint_model(str(checkpoint),torch.device('cpu'))
    with torch.inference_mode(): actual = restored(x)
    torch.testing.assert_close(actual, expected)


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError): create_model('unknown')
