import cv2
import os
import numpy as np
from tqdm import tqdm
# 全部resize成640 * 640

data_path = '/home/penguin/EDisk/聚类学习/聚类图片/'

# train_x = np.fromfile('/home/penguin/EDisk/聚类学习/train_X.bin', dtype=np.uint8)
# train_y = np.fromfile('/home/penguin/EDisk/聚类学习/train_Y.bin', dtype=np.uint8)
# test_x = np.fromfile('/home/penguin/EDisk/聚类学习/train_X.bin', dtype=np.uint8)
# test_y = np.fromfile('/home/penguin/EDisk/聚类学习/train_Y.bin', dtype=np.uint8)
# train_input = np.reshape(train_x, (-1, 3, 640, 640))
# train_labels = train_y
# train_input_flat = np.reshape(test_x, (-1, 1, 3 * 640 * 640))
# test_input = np.reshape(test_x, (-1, 3, 640, 640))
# test_labels = test_y
# test_input_flat = np.reshape(test_x, (-1, 1, 3 * 640 * 640))
#
# np.concatenate((train_input, test_input))
#
# result = [np.concatenate(train_input, test_input), np.concatenate(train_labels, test_labels),
#                 np.concatenate(train_input_flat, test_input_flat)]

# train_x = np.fromfile('/home/penguin/EDisk/聚类学习/train_X.bin', dtype=np.uint8)
# train_input = np.reshape(train_x, (-1, 3, 640, 640))

imgs = []
for i in range(7848):
    img = cv2.imread(os.path.join(data_path, '{}.jpg'.format(i)), -1)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (128, 128))
    if len(imgs) == 0:
        imgs = img.reshape(1, 3, 128, 128)
    else:
        print(i)
        imgs = np.concatenate((imgs, img.reshape(1, 3, 128, 128)), 0)
np.save('/home/penguin/EDisk/聚类学习/train_X.bin', imgs)

# with open('/home/penguin/tttest/test.bin', 'rb') as f:
#     # read whole file in uint8 chunks
#     everything = np.fromfile(f, dtype=np.uint8)
#     images = np.reshape(everything, (-1, 3, 640, 640))
#
# everything