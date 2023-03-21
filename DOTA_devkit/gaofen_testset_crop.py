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
import cv2 as cv

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
             imagesetfile,
             classname,
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
    for d in range(nd):
        bb = BB[d, :].astype(float)

        img = cv.imread(os.path.join(image_dir, image_ids[d] + '.tif'))

        x1, y1, x2, y2, x3, y3, x4, y4 = bb
        xmax = int(max(x1, x2, x3, x4))
        xmin = int(min(x1, x2, x3, x4))
        ymax = int(max(y1, y2, y3, y4))
        ymin = int(min(y1, y2, y3, y4))
        crop_img = img[max(0, ymin):min(img.shape[0], ymax), max(0, xmin):min(img.shape[1], xmax)]
        print(image_ids[d])
        print(bb)
        if crop_img.size == 0:
            print('========================\n')
            continue

        cv.imwrite(os.path.join(out_dir, image_ids[d] + '-' + classname + '-' + str(d) + '.tif'), crop_img)
        f_txt.writelines(image_ids[d] + '-' + classname + '-' + str(d) + '_' + lines_sort[d] + '\n')


def main():
    detpath = r'/home/penguin/Experiments/ReDet/work_dirs/ReDet_re50_refpn_1x_gaofen/Task1_results_nms_5classes/Task1_{:s}.txt'
    imagesetfile = r'/home/penguin/EDisk/gaofen/test/testset.txt'
    image_dir = r'/home/penguin/EDisk/gaofen/test/images/'

    out_dir = '/home/penguin/test/'

    iou_thr = 0.5
    ap_dict = {}

    for classname in CLASSES:
        _out_dir = out_dir + classname + '/'
        _out_txt = out_dir + classname + '.txt'
        voc_eval(detpath, imagesetfile, classname, _out_dir, image_dir, _out_txt, ovthresh=iou_thr, use_07_metric=True)

if __name__ == '__main__':
    main()
