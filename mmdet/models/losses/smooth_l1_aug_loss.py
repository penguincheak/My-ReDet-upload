import torch.nn as nn
from mmdet.core import weighted_smoothl1_aug

from ..registry import LOSSES


@LOSSES.register_module
class SmoothL1AugLoss(nn.Module):

    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1AugLoss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, *args, **kwargs):
        loss_bbox = self.loss_weight * weighted_smoothl1_aug(
            pred, target, weight, beta=self.beta, *args, **kwargs)
        return loss_bbox
