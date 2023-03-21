# --------------------------------------------------------
# dota_evaluation_task1
# Licensed under The MIT License [see LICENSE for details]
# Written by Jian Ding
# --------------------------------------------------------
# -*- coding: utf-8 -*-

"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import matplotlib.pyplot as plt
# import cPickle
import numpy as np
import os
import polyiou
import cv2 as cv
import xml.etree.ElementTree as ET
from functools import partial

CLASSES = (['Vehicle', 'Ship', 'Airplane', 'Court', 'Road'])


def draw_pr(rec, prec, classname):
    plt.figure()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('PR cruve')

    plt.plot(rec.tolist(), prec.tolist())
    plt.savefig('{}_pr.png'.format(classname))

def parse_gt(filename):
    """
    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# cnt 必须是float32 而且得是PointSet (x1, y1)
def drow_box(cnt):
    rotated_box = cv.minAreaRect(cnt)

    box = cv.boxPoints(rotated_box)
    box = np.int0(box)

    return rotated_box, box


def crop2(img, cnt):
    rotated_box, box = drow_box(cnt)

    width = int(rotated_box[1][0])
    height = int(rotated_box[1][1])

    print(width, height)

    if width > height:
        w = width
        h = height
    else:
        w = height
        h = width

    src_pts = box.astype("float32")
    dst_pts = np.array([[w - 1, h - 1],
                        [0, h - 1],
                        [0, 0],
                        [w - 1, 0]], dtype="float32")

    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv.warpPerspective(img, M, (w, h))

    return warped



def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             _annpath,
             out_dir,
             image_dir,
             out_txt,
             # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections results
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    #################################################################################################
    ##### 第一步：获取所有的GT标签信息，存入字典recs中或文件annots.pkl中，便于使用 #####################
    #################################################################################################

    # first load gt
    # if not os.path.isdir(cachedir):
    #   os.mkdir(cachedir)
    # cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # print('imagenames: ', imagenames)
    # if not os.path.isfile(cachefile):
    # load annots
    recs = {}
    ann_recs = {}
    for i, imagename in enumerate(imagenames):
        # print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))
        _ann = parse_gt(_annpath.format(imagename))
        ann_recs[imagename] = [item['name'] for item in _ann]
        # 解析标签文件图像进度条
        # if i % 100 == 0:
        #   print ('Reading annotation for {:d}/{:d}'.format(
        #      i + 1, len(imagenames)) )
        # save
        # print ('Saving cached annotations to {:s}'.format(cachefile))
        # with open(cachefile, 'w') as f:
        #   cPickle.dump(recs, f)
    # else:
    # load
    # with open(cachefile, 'r') as f:
    #   recs = cPickle.load(f)

    #################################################################################################
    ##### 第二步：从字典recs中提取当前类型的GT标签信息，存入字典class_recs中，key为图片名imagename #####
    #################################################################################################
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    name_dict = {}
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        # 求所有非难样本的数量，用于求Recall
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    #################################################################################################
    ##### 第三步：从当前class的result文件中读取结果，并将结果按照confidence从大到小排序 ################
    #####        排序后的结果存在BB和image_ids中                                      ################
    #################################################################################################
    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    # 将每个结果条目中第一个数据，就是图像id,这个image_ids是文件名
    image_ids = [x[0].split('.')[0] for x in splitlines]
    # 提取每个结果的置信度，存入confidence
    confidence = np.array([float(x[1]) for x in splitlines])

    # print('check confidence: ', confidence)
    # 提取每个结果的结果，存入BB
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # 对confidence从大到小排序，获取id
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    # print('check sorted_scores: ', sorted_scores)
    # print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    # 对相应的图像的id进行排序
    image_ids = [image_ids[x] for x in sorted_ind]
    lines_sort = [lines[x] for x in sorted_ind]
    # print('check imge_ids: ', image_ids)
    # print('imge_ids len:', len(image_ids))

    f_txt = open(out_txt, 'w')

    #################################################################################################
    ##### 第四步：对比GT参数和result，计算出IOU，在fp和tp相应位置标记1 #################################
    #################################################################################################
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]

        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps

            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            # 在这里裁剪,jmax是索引
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    name = classname
                    if name_dict.get(name) is None:
                        name_dict[name] = 1
                    else:
                        name_dict[name] += 1

                    img = cv.imread(os.path.join(image_dir, image_ids[d] + '.tif'))

                    cnt = np.array(bb.reshape(4, 2), dtype=np.float32)
                    crop_img = crop2(img, cnt)

                    cv.imwrite(os.path.join(out_dir, image_ids[d] + '-' + name + '-' + str(name_dict[name]) + '.jpg'), crop_img)
                    f_txt.writelines(image_ids[d] + '-' + name + '-' + str(name_dict[name]) + '_' + lines_sort[d] + '\n')

                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

    #################################################################################################
    ##### 第五步：计算ap,rec，prec ###################################################################
    #################################################################################################
    ##difficult用于标记真值个数，prec是precision，rec是recall
    # compute precision recall

    # print('check fp:', fp)
    # print('check tp', tp)
    #
    # print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    print(name_dict)

    f_txt.close()

    return rec, prec, ap



def main():
    detpath = r'/home/penguin/Experiments/ReDet/work_dirs/ReDet_re50_refpn_1x_gaofen/Task1_results_nms/Task1_{:s}.txt'
    annopath = r'/home/penguin/EDisk/gaofen/test/labelTxt/{:s}.txt'  # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    class_ann = r'/home/penguin/EDisk/gaofen/test/labelTxt/{:s}.txt'
    imagesetfile = r'/home/penguin/EDisk/gaofen/test/testset.txt'
    image_dir = r'/home/penguin/EDisk/gaofen/test/images/'

    out_dir = '/home/penguin/test/'

    iou_thr = 0.5
    ap_dict = {}

    for classname in CLASSES:
        _out_dir = out_dir + classname + '/'
        _out_txt = out_dir + classname + '.txt'
        rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, classname, class_ann, _out_dir, image_dir, _out_txt, ovthresh=iou_thr, use_07_metric=True)
        ap_dict[classname] = ap
        print('ap_dict:iou_thr %f\n' % iou_thr, ap_dict)
        mean = 0
        for k, v in ap_dict.items():
            mean = mean + v
        mean = mean / float(len(ap_dict))
    print('AP50: {:.2f}'.format(mean))

if __name__ == '__main__':
    main()
