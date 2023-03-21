import os
import numpy as np
import mmcv
import re
import cv2

from tqdm import tqdm
import cv2 as cv

img_root = r'/home/penguin/test/dior/images/'
label_root = r'/home/penguin/test/dior/label_txt/'
save_root = r'/home/penguin/test_gt/dior'
name_dic = {}
h = 0
w = 0

# class_name = ('plane', 'baseball-diamond',
#                 'bridge', 'ground-track-field',
#                 'small-vehicle', 'large-vehicle',
#                 'ship', 'tennis-court',
#                 'basketball-court', 'storage-tank',
#                 'soccer-ball-field', 'roundabout',
#                 'harbor', 'swimming-pool',
#                 'helicopter', 'container-crane')
class_name = ('airplane', 'airport', 'baseballfield',
               'basketballcourt', 'bridge', 'chimney',
               'dam', 'Expressway-Service-area',
               'Expressway-toll-station', 'harbor',
               'golffield', 'groundtrackfield', 'overpass',
               'ship', 'stadium', 'storagetank',
               'tenniscourt', 'trainstation', 'vehicle',
               'windmill')

dota15_colormap = [
    (54, 67, 244),
    (99, 30, 233),
    (176, 39, 156),
    (183, 58, 103),
    (181, 81, 63),
    (243, 150, 33),
    (212, 188, 0),
    (136, 150, 0),
    (80, 175, 76),
    (74, 195, 139),
    (57, 220, 205),
    (59, 235, 255),
    (0, 152, 255),
    (34, 87, 255),
    (72, 85, 121),
    (139, 125, 96),
    (43, 58, 156),
    (69, 73, 123),
    (187, 12, 39),
    (78, 86, 90)]

def draw_poly_detections(img, detections, class_names, scale, threshold=0.2, putText=True,showStart=False, colormap=None):
    """

    :param img:
    :param detections:
    :param class_names:
    :param scale:
    :param cfg:
    :param threshold:
    :return:
    """
    import pdb
    import cv2
    import random
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    color_white = (255, 255, 255)

    for j, name in enumerate(class_names):
        if colormap is None:
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        else:
            color = colormap[j]
        try:
            dets = detections[j]
        except:
            pdb.set_trace()
        for det in dets:
            bbox = det[:8] * scale
            score = det[-1]
            if score < threshold:
                continue
            bbox = list(map(int, bbox))
            if showStart:
                cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
            for i in range(3):
                cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]), color=color, thickness=2,lineType=cv2.LINE_AA)
            cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2,lineType=cv2.LINE_AA)
            if putText:
                cv2.putText(img, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                            color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return img


for i in tqdm(os.listdir(label_root)):
    f = open(os.path.join(label_root, i))
    img = cv.imread(os.path.join(img_root, i.replace('txt', 'jpg')))
    detections = []
    for num in range(len(class_name)):
        detections.append([])
    for j in f.readlines():
        if j.strip() == 'imagesource:GoogleEarth' or re.match(r'gsd:', j.strip()) != None:
            continue
        boxes = j.strip().split(' ')[:8]
        index = class_name.index(j.strip().split(' ')[8])
        boxes = [float(i) for i in boxes]
        boxes.append(1)
        detections[index].append(boxes)
        # x1, y1, x2, y2, x3, y3, x4, y4, name, diff = j.strip().split(' ')
        # x1, y1, x2, y2, x3, y3, x4, y4 = [float(m) for m in [x1, y1, x2, y2, x3, y3, x4, y4]]
        # if name_dic.get(name) is None:
        #     name_dic[name] = 1
        # else:
        #     name_dic[name] += 1
        # xmax = int(max(x1, x2, x3, x4))
        # xmin = int(min(x1, x2, x3, x4))
        # ymax = int(max(y1, y2, y3, y4))
        # ymin = int(min(y1, y2, y3, y4))
        # h += xmax - xmin
        # w += ymax - ymin
        # temp = img[max(0, ymin):min(img.shape[0], ymax), max(0, xmin):min(img.shape[1], xmax)]
        # cv.imwrite(os.path.join(save_root, name+'-'+str(name_dic[name])+'.png'), temp)
    detections = [np.array(i) for i in detections]
    img = draw_poly_detections(os.path.join(img_root, i.replace('txt', 'jpg')), detections, class_name, scale=1, threshold=0.2,
                                   colormap=dota15_colormap)
    cv2.imwrite(os.path.join(save_root, i.replace('txt', 'jpg')), img)

total = 0
for i in name_dic.keys():
    total += name_dic[i]
print(name_dic)