import torch
import pytest
from pipe_counter.detection import create_detector, configure_detector, load_detector, pipe_scores, validate_detector_initialization
from torchvision.ops.misc import FrozenBatchNorm2d


def test_fcos_offline_reload_preserves_structure_limits_and_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.hub,'download_url_to_file',lambda *a,**k:pytest.fail('Unexpected download'))
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model=create_detector(96,False,architecture='FCOSResNet50FPN').eval()
    configure_detector(model,.3,.01,2000)
    assert model.head.classification_head.cls_logits.out_channels==2
    assert model.detections_per_img==2000 and model.topk_candidates==10000
    assert isinstance(model.backbone.body.bn1,FrozenBatchNorm2d)
    assert not any(isinstance(m,torch.nn.BatchNorm2d) for m in model.backbone.modules())
    fixture=torch.rand(3,64,80)
    with torch.inference_mode():expected=model([fixture])[0]
    path=tmp_path/'fcos.pt'
    torch.save({'model':model.state_dict(),'threshold':.3,'config':{
        'architecture':'FCOSResNet50FPN','image_size':96,'box_nms_threshold':.3,
        'minimum_score':.01,'max_detections':2000,'topk_candidates':10000}},path)
    reloaded,_=load_detector(path,torch.device('cpu'))
    with torch.inference_mode():actual=reloaded([fixture])[0]
    for key in ['boxes','scores','labels']:assert torch.equal(expected[key],actual[key])


def test_background_is_not_counted_and_cross_architecture_initialization_is_rejected():
    result={'scores':torch.tensor([.95,.8,.2]),'labels':torch.tensor([0,1,1])}
    assert torch.equal(pipe_scores(result),torch.tensor([.8,.2]))
    parent={'config':{'architecture':'FCOSResNet50FPN','geometry_overrides_sha256':'g'},
        'development_data':{'sha256':['train','val']}}
    dev={'sha256':['train','val']}
    validate_detector_initialization(parent,dev,'g',architecture='FCOSResNet50FPN')
    with pytest.raises(ValueError):validate_detector_initialization(parent,dev,'g')
