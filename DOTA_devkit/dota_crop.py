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

CLASSES = (['plane', 'baseball-diamond',
                'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle',
                'ship', 'tennis-court',
                'basketball-court', 'storage-tank',
                'soccer-ball-field', 'roundabout',
                'harbor', 'swimming-pool',
                'helicopter', 'container-crane'])

mmax = 0
mmin = 10000


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


def crop2(cnt):
    global mmax, mmin
    rotated_box, box = drow_box(cnt)

    width = int(rotated_box[1][0])
    height = int(rotated_box[1][1])

    if width * height > mmax:
        mmax = width * height
        print("mmax = " + str(mmax) + '\n')

    if width * height < mmin:
        mmin = width * height
        print("mmin = " + str(mmin) + '\n')





def voc_eval(annopath,
             imagesetfile,
             classname):
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
    for i, imagename in enumerate(imagenames):
        # print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))
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
        for i in range(len(bbox)):
            bb = bbox[i]
            cnt = np.array(bb.reshape(4, 2), dtype=np.float32)
            crop_img = crop2(cnt)


def main():
    annopath = r'/home/penguin/DataSet/dota/train/labelTxt/{:s}.txt'  # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    imagesetfile = r'/home/penguin/DataSet/dota/train/trainset.txt'

    for classname in CLASSES:
        voc_eval(annopath, imagesetfile, classname)
    print("mmax:" + str(mmax) + '\n' + "mmin:" + str(mmin))

if __name__ == '__main__':
    main()
