import os

data_path = '/home/penguin/EDisk/聚类学习/分类图片/'
ccount = 0
for i in os.listdir(data_path):
    if i == '.DS_Store':
        continue
    img_path = os.path.join(data_path, i)
    for j in os.listdir(img_path):
        if j == '.DS_Store':
            continue
        os.rename(os.path.join(img_path, j), os.path.join(img_path, '{}.jpg'.format(ccount)))
        ccount += 1