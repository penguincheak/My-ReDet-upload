import os
import shutil
import re

from tqdm import tqdm
import cv2 as cv

img_root = r'/home/penguin/可视化/images/'
label_root = r'/home/penguin/可视化/label/'
save_root = r'/home/penguin/可视化/'
name_dic = {}
h = 0
w = 0
for i in tqdm(os.listdir(label_root)):
    f = open(os.path.join(label_root, i))
    img = cv.imread(os.path.join(img_root, i.replace('txt', 'png')))

    for j in f.readlines():
        if j.strip() == 'imagesource:GoogleEarth' or re.match('gsd', j) != None:
            continue
        x1, y1, x2, y2, x3, y3, x4, y4, name, _ = j.strip().split(' ')
        x1, y1, x2, y2, x3, y3, x4, y4 = [float(m) for m in [x1, y1, x2, y2, x3, y3, x4, y4]]
        if name_dic.get(name) is None:
            name_dic[name] = 1
        else:
            name_dic[name] += 1
        xmax = int(max(x1, x2, x3, x4))
        xmin = int(min(x1, x2, x3, x4))
        ymax = int(max(y1, y2, y3, y4))
        ymin = int(min(y1, y2, y3, y4))
        h += xmax - xmin
        w += ymax -ymin
        temp = img[max(0, ymin):min(img.shape[0], ymax), max(0, xmin):min(img.shape[1], xmax)]
        cv.imwrite(os.path.join(save_root, name+'-'+str(name_dic[name])+'.jpg'), temp)
total = 0
for i in name_dic.keys():
    total += name_dic[i]
print(name_dic)