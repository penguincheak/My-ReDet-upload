import cv2
import os
import numpy as np
from tqdm import tqdm
# 全部resize成640 * 640

data_path = '/home/penguin/EDisk/聚类学习/分类图片/'
label_name = {'补给舰': 0, '登陆舰': 1, '航空母舰': 2, '护卫舰': 3, '两栖攻击舰': 4, '猎潜舰': 5, '驱逐舰': 6, '运输舰': 7, '侦测舰': 8}
label_index = np.zeros(7848)

ccount = 0
for i in os.listdir(data_path):
    if i == '.DS_Store':
        continue
    img_path = os.path.join(data_path, i)
    for j in os.listdir(img_path):
        if j == '.DS_Store':
            continue
        j = j[0: j.rfind('.')]
        label_index[int(j)] = label_name[i]
        ccount += 1
print(ccount)
np.save('/home/penguin/EDisk/聚类学习/train_Y.npy', label_index.T)