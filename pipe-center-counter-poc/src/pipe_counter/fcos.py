"""Anchor-free detector comparison with consistent pretrained/offline structure."""
import math
import torch
from torch import nn
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
from torchvision.ops.misc import FrozenBatchNorm2d

ARCHITECTURE = 'FCOSResNet50FPN'
WEIGHTS_NAME = 'fcos_resnet50_fpn_coco-99b0c9b7.pth'
WEIGHTS_SHA256 = '99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7'


def _frozen_batch_norm(module):
    # Torchvision chooses different BN types depending on pretrained=True.
    # Construct the pretrained layout for both training and offline reload.
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            frozen.load_state_dict({k:v for k,v in child.state_dict().items() if k!='num_batches_tracked'})
            setattr(module,name,frozen)
        else:
            _frozen_batch_norm(child)


def create_fcos(image_size=960, pretrained=False):
    model=fcos_resnet50_fpn(weights=None,weights_backbone=None,
        min_size=image_size,max_size=image_size,score_thresh=.01,nms_thresh=.3,
        detections_per_img=2000,topk_candidates=10000)
    _frozen_batch_norm(model.backbone)
    if pretrained:
        model.load_state_dict(FCOS_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=False,check_hash=True))
    head=model.head.classification_head
    old=head.cls_logits
    head.num_classes=2
    head.cls_logits=nn.Conv2d(old.in_channels,head.num_anchors*2,kernel_size=3,padding=1)
    nn.init.normal_(head.cls_logits.weight,std=.01)
    nn.init.constant_(head.cls_logits.bias,-math.log(99.))
    for name,param in model.backbone.body.named_parameters():
        param.requires_grad_(name.startswith(('layer2.','layer3.','layer4.')))
    return model
