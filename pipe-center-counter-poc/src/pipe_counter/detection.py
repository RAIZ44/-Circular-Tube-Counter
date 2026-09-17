"""Bounding-box detector experiment using the same frozen pipe manifests."""
from __future__ import annotations

import random
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from .utils import read_jsonl

ARCHITECTURE = 'FasterRCNNResNet50FPNV2'
DETECTOR_ARCHITECTURES = (ARCHITECTURE, 'FCOSResNet50FPN')
WEIGHTS_NAME = 'fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'
WEIGHTS_SHA256 = 'dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf'


def convert_boxes(boxes, width, height):
    xywh = np.asarray(boxes, dtype=np.float32).reshape(-1,4)
    converted = np.empty_like(xywh)
    converted[:,:2] = xywh[:,:2] - xywh[:,2:]/2
    converted[:,2:] = xywh[:,:2] + xywh[:,2:]/2
    converted[:,[0,2]] = np.clip(converted[:,[0,2]],0,width)
    converted[:,[1,3]] = np.clip(converted[:,[1,3]],0,height)
    if not np.isfinite(converted).all() or np.any(converted[:,2:] <= converted[:,:2]):
        raise ValueError('Invalid/degenerate annotations: review before detector training')
    return torch.from_numpy(converted)


class DetectionDataset(Dataset):
    def __init__(self, manifest, augment=False, geometry_overrides=None):
        self.records = read_jsonl(manifest)
        self.augment = augment
        self.geometry_overrides = {}
        if geometry_overrides is not None:
            self.geometry_overrides = json.loads(Path(geometry_overrides).read_text(encoding='utf-8'))['images']
        if not self.records:
            raise ValueError('Empty manifest')
        # Audit every annotation before launching an expensive run. Do not drop labels.
        for record in self.records:
            convert_boxes(record['boxes'],record['width'],record['height'])
            if record.get('sha256') in self.geometry_overrides:
                digest=hashlib.sha256(Path(record['image_path']).read_bytes()).hexdigest()
                if digest != record['sha256']:
                    raise ValueError('Audited image bytes changed; geometry correction refused')

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        record = self.records[index]
        image = cv2.imread(record['image_path'],cv2.IMREAD_COLOR)
        if image is None: raise RuntimeError(record['image_path'])
        height,width = image.shape[:2]
        boxes = boxes_for_decoded_image(record,width,height,self.geometry_overrides)
        if self.augment and random.random()<.5:
            image=np.ascontiguousarray(image[:,::-1])
            boxes[:,[0,2]]=width-boxes[:,[2,0]]
        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        tensor=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()/255.
        return tensor, {'boxes':boxes,'labels':torch.ones(len(boxes),dtype=torch.int64),
                        'image_id':torch.tensor(index,dtype=torch.int64)}


def boxes_for_decoded_image(record,width,height,overrides):
    if (width,height)==(record['width'],record['height']):
        return convert_boxes(record['boxes'],width,height)
    correction=overrides.get(record.get('sha256'))
    if correction is None:
        raise ValueError(f"Image size differs from annotation metadata: {record['image_path']}; decoded={(width,height)}, metadata={(record['width'],record['height'])}")
    if correction['annotation_size'] != [record['width'],record['height']] or correction['decoded_size'] != [width,height]:
        raise ValueError('Geometry correction dimensions do not match the audited record')
    if correction['label_count'] != len(record['boxes']):
        raise ValueError('Annotation count changed since geometry audit')
    if correction['operation']=='labels_ccw90' and (width,height)==(record['height'],record['width']):
        corrected=[[y,record['width']-x,h,w] for x,y,w,h in record['boxes']]
    elif correction['operation']=='empty_labels_metadata_only' and not record['boxes']:
        corrected=[]
    else:
        raise ValueError('Unsupported or unverified geometry correction')
    return convert_boxes(corrected,width,height)


def collate_detection(batch):
    return tuple(zip(*batch))


def create_detector(image_size=640,pretrained=False,architecture=ARCHITECTURE):
    if architecture=='FCOSResNet50FPN':
        from .fcos import create_fcos
        return create_fcos(image_size,pretrained)
    if architecture!=ARCHITECTURE:
        raise ValueError('Unsupported detector architecture')
    model=fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None,min_size=image_size,max_size=image_size,
        rpn_pre_nms_top_n_train=8000,rpn_pre_nms_top_n_test=8000,
        rpn_post_nms_top_n_train=2000,rpn_post_nms_top_n_test=4000,
        rpn_batch_size_per_image=512,box_batch_size_per_image=512,
        box_positive_fraction=.5,box_detections_per_img=2000,
        box_score_thresh=.01,box_nms_thresh=.5)
    # Same number/ratios of anchors preserves the pretrained RPN tensor shapes;
    # smaller spatial sizes fit dense ends better than generic COCO sizes.
    model.rpn.anchor_generator=AnchorGenerator(((16,),(32,),(64,),(128,),(256,)),((.5,1.,2.),)*5)
    features=model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(features,2)
    # Keep the same frozen early ResNet layers for training and offline reload.
    for name,param in model.backbone.body.named_parameters():
        param.requires_grad_(name.startswith(('layer2.','layer3.','layer4.')))
    return model


def load_detector(path,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint['config']['architecture'] not in DETECTOR_ARCHITECTURES:
        raise ValueError('Not a detector checkpoint')
    model=create_detector(checkpoint['config']['image_size'],pretrained=False,architecture=checkpoint['config']['architecture'])
    model.load_state_dict(checkpoint['model'])
    config=checkpoint['config']
    if torch.device(device).type=='cuda':
        if 'cudnn_allow_tf32' in config:
            torch.backends.cudnn.allow_tf32=bool(config['cudnn_allow_tf32'])
        if 'cuda_matmul_allow_tf32' in config:
            torch.backends.cuda.matmul.allow_tf32=bool(config['cuda_matmul_allow_tf32'])
    configure_detector(model,float(config.get('box_nms_threshold',.5)),float(config.get('minimum_score',.01)),int(config.get('max_detections',2000)))
    if 'topk_candidates' in config and hasattr(model,'topk_candidates'):
        model.topk_candidates=int(config['topk_candidates'])
    return model.to(device).eval(),checkpoint


def configure_detector(model,nms_threshold,minimum_score=.01,max_detections=2000):
    target=model.roi_heads if hasattr(model,'roi_heads') else model
    target.nms_thresh=nms_threshold
    target.score_thresh=minimum_score
    target.detections_per_img=max_detections


def pipe_scores(prediction):
    # One-stage classifiers can return background index 0. Never count it.
    return prediction['scores'][prediction['labels']==1]


def validate_detector_initialization(parent,development,geometry_hash,architecture=ARCHITECTURE):
    if architecture not in DETECTOR_ARCHITECTURES or parent.get('config',{}).get('architecture')!=architecture:
        raise ValueError('Initial checkpoint is not the matching detector architecture')
    prior=parent.get('development_data')
    if not prior or not prior.get('sha256') or not set(prior['sha256']).issubset(set(development['sha256'])):
        raise ValueError('Initial checkpoint has unknown or incompatible development data')
    if parent['config'].get('geometry_overrides_sha256')!=geometry_hash:
        raise ValueError('Geometry changed since initialization; review before continuing')
