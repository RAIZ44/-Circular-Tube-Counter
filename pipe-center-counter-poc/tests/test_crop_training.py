import json
import cv2
import numpy as np
import pytest
from pipe_counter.data import crop_to_window, PipeCenterDataset


def test_crop_preserves_centers_and_excludes_other_side_of_seam():
    image = np.zeros((40,80,3), np.uint8)
    boxes = [[10.,10.,8.,6.],[39.5,20.,4.,5.],[40.,20.,4.,5.],[70.,20.,4.,5.]]
    left, a = crop_to_window(image, boxes, (0,0,40,40))
    right, b = crop_to_window(image, boxes, (40,0,80,40))
    assert left.shape == right.shape == (40,40,3)
    assert a == boxes[:2]
    assert b == [[0.,20.,4.,5.],[30.,20.,4.,5.]]
    assert len(a)+len(b) == len(boxes)
    assert boxes[2][0] == 40.  # Never mutate source labels.
    assert crop_to_window(image, boxes, (0,30,10,40))[1] == []
    with pytest.raises(ValueError):
        crop_to_window(image,boxes,(0,0,81,40))


def test_validation_is_never_cropped_even_with_crop_options(tmp_path, monkeypatch):
    image_path = tmp_path/'image.jpg'
    cv2.imwrite(str(image_path), np.zeros((64,64,3), np.uint8))
    manifest = tmp_path/'val.jsonl'
    manifest.write_text(json.dumps({'image_path':str(image_path),'boxes':[[10,10,6,6],[50,50,6,6]]})+'\n')
    dataset = PipeCenterDataset(manifest, 64, stride=2, augment=False, crop_probability=1)
    monkeypatch.setattr('pipe_counter.data.crop_to_window', lambda *args: pytest.fail('Validation was cropped'))
    assert dataset[0]['count'].item() == 2
    with pytest.raises(ValueError):
        PipeCenterDataset(manifest, crop_probability=2)
