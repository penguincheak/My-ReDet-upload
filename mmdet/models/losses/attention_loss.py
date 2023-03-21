import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2

# mmax:1426029
# mmin:16

def make_r_gt_mask(fet_h, fet_w, img_h, img_w, gtboxes):
    gtboxes = np.reshape(gtboxes, [-1, 6])  # [x, y, w, h, theta, label]

    areas = gtboxes[:, 2] * gtboxes[:, 3]
    arg_areas = np.argsort(-1 * areas)  # sort from large to small
    gtboxes = gtboxes[arg_areas]

    fet_h, fet_w = int(fet_h), int(fet_w)
    mask = np.zeros(shape=[fet_h, fet_w], dtype=np.int32)
    for a_box in gtboxes:
        # print(a_box)
        box = cv2.boxPoints(((a_box[0], a_box[1]), (a_box[2], a_box[3]), a_box[4]))
        box = np.reshape(box, [-1, ])
        label = a_box[-1]
        new_box = []
        for i in range(8):
            if i % 2 == 0:
                x = box[i]
                new_x = int(x * fet_w / float(img_w))
                new_box.append(new_x)
            else:
                y = box[i]
                new_y = int(y*fet_h/float(img_h))
                new_box.append(new_y)

        new_box = np.int0(new_box).reshape([4, 2])
        color = int(label)
        # print(type(color), color)
        cv2.fillConvexPoly(mask, new_box, color=color)
    # print (mask.dtype)
    return mask

def dice_loss(target, predictive, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

class AttentionLoss(nn.Module):
    def forward(self, img_batch_shape, attention_mask, bboxs):

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

            mask_loss = []
            # 对不同尺度的att_heatmaps操作
            for id in range(len(attention_mask)):

                attention_map = attention_mask[id][j, 0, :, :]

                level_bbox_annotation = bbox_annotation.clone()

                attention_h, attention_w = attention_map.shape

                if level_bbox_annotation.shape[0]:
                    level_bbox_annotation[:, 0] *= attention_w / w
                    level_bbox_annotation[:, 1] *= attention_h / h
                    level_bbox_annotation[:, 2] *= attention_w / w
                    level_bbox_annotation[:, 3] *= attention_h / h

                mask_gt = torch.zeros(attention_map.shape)
                mask_gt = mask_gt.cuda()

                # 这一步生成mask，需要改成OBB
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