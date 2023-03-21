import os

path_label = '/home/penguin/Task1_results_nms/'
files = os.listdir(path_label)  # 路径
# train_val = open('/home/penguin/Experiments/ReDet/data/gaofen-challenge/train/testset.txt','w')
path = '/home/penguin/result_project/'
i = 0
for file in files:
    print(i)
    i += 1
    f = open(path_label + '/' + file)  # 返回一个文件对象
    cls_name = file[file.rfind('_', 1) + 1: file.rfind('.', 1)]
    line = f.readline()
    while(line):
        # 写入格式：置信度 8个坐标点
        # 读取格式：图片名 置信度 8个坐标点
        line = line.split(' ')
        img_name = line[0]
        confidence, x0, y0, x1, y1, x2, y2, x3, y3 = line[1: len(line)]
        y3 = y3[0: y3.rfind('\n')]
        conference = confidence + ' ' + x0 + ' ' + y0 + ' ' + x1 + ' ' + y1 + ' ' \
                     + x2 + ' ' + y2 + ' ' + x3 + ' ' + y3 + ' ' + cls_name
        res = open(path + img_name + '.txt', 'a')
        res.write(conference + '\n')
        res.close()

        line = f.readline()
    f.close()

