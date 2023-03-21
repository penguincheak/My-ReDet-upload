import torch.nn as nn
from mmdet.core import iou_smoothl1_loss

from ..registry import LOSSES


@LOSSES.register_module
class IoUSmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, loss_weight=1.0):
        super(IoUSmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, bbox_par, *args, **kwargs):
        loss_bbox = self.loss_weight * iou_smoothl1_loss(
            pred, target, weight, beta=self.beta, bbox_par=bbox_par, *args, **kwargs)
        return loss_bbox
