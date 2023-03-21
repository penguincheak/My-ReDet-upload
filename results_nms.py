import cv2
import math
import mmcv
import numpy as np
import os
import pdb
from mmcv import Config

import DOTA_devkit.polyiou as polyiou
from mmdet.apis import init_detector, inference_detector, draw_poly_detections
from mmdet.datasets import get_dataset

def py_cpu_nms_poly_fast_np(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(dets.shape[0]):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                           dets[i][2], dets[i][3],
                                           dets[i][4], dets[i][5],
                                           dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou

        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

if __name__ == '__main__':
    res_path = "/home/penguin/Experiments/ReDet/work_dirs/ReDet_re50_refpn_1x_gaofen/Task1_results_nms_34classes/"
    det_path = "/home/penguin/Task_1_nms_34classes/"
    for file in os.listdir(res_path):
        f = open(res_path + file, 'r')
        w = open(det_path + file, 'a')
        res = []
        line = f.readline()
        while(line):
            line = line.split(' ')
            line[9] = line[9][0: line[9].rfind('\n')]
            det = []
            for i in range(0, 10):
                det.append(float(line[i]))
            res.append(det)
            line = f.readline()
        res = np.array(res)
        keep = py_cpu_nms_poly_fast_np(np.c_[res[:, 2:], res[:, 1]], 0.7)
        res = res[keep]
        len = res.shape[0]
        count = 0
        for i in range(0,len):
            count += 1
            line = res[i]
            s = str(int(line[0])) + " "
            for j in range(1, line.shape[0]):
                if j < line.shape[0] - 1:
                    s = s + str(line[j]) + " "
                else:
                    s = s + str(line[j])
            w.write(s + '\n')
            print(count)
        f.close()
        w.close()
