import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# dota1.5
# mmax:1426029
# mmin:16

# dota1
# mmax:2884902.0
# mmin:9.0

# hrsc2016
# mmax:456383
# mmin:1049

# NWPU-VHR10
# mmax:167535.0
# mmin:510.0

def dice_loss(target, predictive, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

class AttentionLoss_SSP(nn.Module):
    def forward(self, img_batch_shape, attention_mask, bboxs, alpha, beta):

        h, w = img_batch_shape[0], img_batch_shape[1]

        mask_losses = []

        batch_size = len(bboxs)
        for j in range(batch_size):

            bbox_annotation = bboxs[j]
            # bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            # 判断bbox是否超出图片边界
            cond1 = torch.le(bbox_annotation[:, 0], w)
            cond2 = torch.le(bbox_annotation[:, 1], h)
            cond3 = torch.le(bbox_annotation[:, 2], w)
            cond4 = torch.le(bbox_annotation[:, 3], h)
            cond = cond1 * cond2 * cond3 * cond4

            bbox_annotation = bbox_annotation[cond, :]

            if bbox_annotation.shape[0] == 0:
                mask_losses.append(torch.tensor(0).float().cuda())
                continue
            # 计算bbox的面积，也即尺度
            bbox_area = torch.abs((bbox_annotation[:, 2] - bbox_annotation[:, 0]) * (
                        bbox_annotation[:, 3] - bbox_annotation[:, 1]))

            mask_loss = []
            # 对不同尺度的att_heatmaps操作
            for id in range(len(attention_mask)):
                for channel in range(attention_mask[id].size(1)):

                    attention_map = attention_mask[id][j, channel, :, :]

                    # 这个尺度的判断需要考虑清楚，因为不同物体的比例不一样
                    min_area = (2 ** (id + alpha)) ** 2
                    max_area = (2 ** (id + alpha) * beta) ** 2

                    # 根据尺度范围对gt进行筛选，选出每个层应该要注意的目标
                    level_bbox_indice1 = torch.ge(bbox_area, min_area)
                    level_bbox_indice2 = torch.le(bbox_area, max_area)

                    level_bbox_indice = level_bbox_indice1 * level_bbox_indice2

                    level_bbox_annotation = bbox_annotation[level_bbox_indice, :].clone()

                    # level_bbox_annotation = bbox_annotation.clone()

                    attention_h, attention_w = attention_map.shape

                    if level_bbox_annotation.shape[0]:
                        level_bbox_annotation[:, 0] *= attention_w / w
                        level_bbox_annotation[:, 1] *= attention_h / h
                        level_bbox_annotation[:, 2] *= attention_w / w
                        level_bbox_annotation[:, 3] *= attention_h / h

                    mask_gt = torch.zeros(attention_map.shape)
                    mask_gt = mask_gt.cuda()

                    for i in range(level_bbox_annotation.shape[0]):
                        x1 = max(int(level_bbox_annotation[i, 0]), 0)
                        y1 = max(int(level_bbox_annotation[i, 1]), 0)
                        x2 = min(math.ceil(level_bbox_annotation[i, 2]) + 1, attention_w)
                        y2 = min(math.ceil(level_bbox_annotation[i, 3]) + 1, attention_h)

                        mask_gt[y1:y2, x1:x2] = 1

                    mask_gt = mask_gt[mask_gt >= 0]
                    mask_predict = attention_map[attention_map >= 0]

                    mask_loss.append(0.5 * F.binary_cross_entropy(mask_predict, mask_gt) +
                                     0.5 * dice_loss(mask_predict, mask_gt))
            mask_losses.append(torch.stack(mask_loss).mean())

        return torch.stack(mask_losses).mean(dim=0, keepdim=True)