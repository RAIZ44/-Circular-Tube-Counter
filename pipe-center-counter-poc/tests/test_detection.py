import json
import cv2
import numpy as np
import pytest
import torch
from pipe_counter.detection import convert_boxes, DetectionDataset, create_detector, boxes_for_decoded_image, validate_detector_initialization, ARCHITECTURE


def test_box_conversion_clips_edges_without_losing_or_reordering_labels():
    boxes=convert_boxes([[5,5,20,20],[40,20,10,4]],50,30)
    assert torch.equal(boxes,torch.tensor([[0.,0.,15.,15.],[35.,18.,45.,22.]]))
    assert convert_boxes([],50,30).shape==(0,4)
    with pytest.raises(ValueError):convert_boxes([[60,20,2,2]],50,30)
    with pytest.raises(ValueError):convert_boxes([[5,5,0,2]],50,30)


def test_detection_input_is_rgb_with_valid_boxes_and_empty_support(tmp_path):
    path=tmp_path/'red.png'
    cv2.imwrite(str(path),np.full((20,40,3),(0,0,255),dtype=np.uint8))
    manifest=tmp_path/'val.jsonl'
    record={'image_path':str(path),'width':40,'height':20,'boxes':[]}
    manifest.write_text(json.dumps(record)+'\n')
    image,target=DetectionDataset(manifest)[0]
    assert image.shape==(3,20,40)
    assert image[0].eq(1).all() and image[2].eq(0).all()
    assert target['boxes'].shape==(0,4) and target['labels'].dtype==torch.int64


def test_detector_can_count_dense_scenes_and_load_without_download(monkeypatch):
    monkeypatch.setattr(torch.hub,'download_url_to_file',lambda *a,**k:pytest.fail('Unexpected download'))
    model=create_detector(128,pretrained=False)
    assert model.roi_heads.detections_per_img==2000
    assert model.rpn._post_nms_top_n['testing']==4000
    assert model.roi_heads.box_predictor.cls_score.out_features==2
    assert model.rpn.anchor_generator.sizes[0]==(16,)
    assert model.transform.min_size==(128,)


def test_geometry_correction_requires_audited_identity_and_preserves_count():
    record={'sha256':'known','image_path':'example.jpg','width':100,'height':200,'boxes':[[20,50,10,20],[80,150,10,20]]}
    correction={'annotation_size':[100,200],'decoded_size':[200,100],
                'label_count':2,'operation':'labels_ccw90'}
    result=boxes_for_decoded_image(record,200,100,{'known':correction})
    assert torch.equal(result,torch.tensor([[40.,75.,60.,85.],[140.,15.,160.,25.]]))
    assert len(result)==len(record['boxes'])
    assert record['boxes'][0]==[20,50,10,20]
    with pytest.raises(ValueError):boxes_for_decoded_image(record,200,100,{})
    with pytest.raises(ValueError):boxes_for_decoded_image(record,201,100,{'known':correction})
    record['boxes']=[]
    with pytest.raises(ValueError):boxes_for_decoded_image(record,200,100,{'known':correction})


def test_detector_initialization_rejects_unknown_data_or_changed_geometry():
    parent={'config':{'architecture':ARCHITECTURE,'geometry_overrides_sha256':'same'},
            'development_data':{'sha256':['train','validation']}}
    current={'sha256':['train','validation']}
    validate_detector_initialization(parent,current,'same')
    with pytest.raises(ValueError):validate_detector_initialization(parent,{'sha256':['train']},'same')
    with pytest.raises(ValueError):validate_detector_initialization(parent,current,'changed')
    with pytest.raises(ValueError):validate_detector_initialization({'config':{'architecture':ARCHITECTURE}},current,'same')
