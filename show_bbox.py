# -*- coding: UTF-8 -*-
'''
将测试生成的txt文件，把文件中对应的box的坐标画回原图
'''
import cv2
import numpy as np
import os
import re

# filelist = glob.glob('./*.txt')
#     for filename in filelist:
#         # basename = osp.basename(filename)
#         with open(filename, 'r') as fread:
#             lines = fread.readlines()
#             nplines = []
#             # read lines
#             for line in lines:
#                 line = line.split()
#                 npline = np.array(line[:8], dtype=np.float32).astype(np.int32)
#                 nplines.append(npline[np.newaxis])
#             nplines = np.concatenate(nplines, 0).reshape(-1, 4, 2)


def drawBBox(txt_path, img_path, save_path):
    global img_id
    img_id = "00101785"  # 换成第一张图片id
    with open(txt_path, 'r')as fp:
        while (1):
            line = fp.readline()
            if not line:
                print("txt is over!!!")
                break
            if line.strip() == 'imagesource:GoogleEarth' or re.match(r'gsd:', line.strip()) != None:
                continue
            str1 = line.split(" ")
            # pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
            # pts = pts.reshape((-1, 1, 2))  # 高、宽、通道数   cv2.resize():宽、高
            # cv.polylines(img, pts=pts, isClosed=True, color=(255, 255, 255), thickness=3)
            boxes = str1[2:]
            # pts = np.array([[int(float(str1[2].strip())), int(float(str1[3].strip()))],
            #                 [int(float(str1[4].strip())), int(float(str1[5].strip()))],
            #                 [int(float(str1[6].strip())), int(float(str1[7].strip()))],
            #                 [int(float(str1[8].strip())), int(float(str1[9].strip()))]
            #                 ], np.int32)
            # pts = pts.reshape((-1, 1, 2))
            if str1[0] != img_id or img_id == "00101785":
                img = cv2.imread(img_path + str1[0] + ".png")
            else:
                # 换成你自己的类别
                img = cv2.imread(save_path + str1[0] + ".png")
            for i in range(3):
                cv2.line(img, (int(float(boxes[i*2].strip())), int(float(boxes[i*2+1].strip()))),
                         (int(float(boxes[(i+1)*2].strip())), int(float(boxes[(i+1)*2+1].strip()))),
                         color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.line(img, (int(float(boxes[6].strip())), int(float(boxes[7].strip()))),
                     (int(float(boxes[0].strip())), int(float(boxes[1].strip()))),
                     color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            # img = cv2.polylines(img, pts, True, (0, 0, 255), 10)
            img_id = str1[0]
            # import pdb
            # pdb.set_trace()
            cv2.imwrite(save_path + img_id + ".png", img)
            print(str1[0] + ".png is save....OK!!!")


if __name__ == '__main__':
    # txt存放的路径
    txt_path = "/home/penguin/可视化/label/P0695.txt"
    # 原图片路径
    img_path = "/home/penguin/可视化/images/"
    # 画出来的图片保存的路径
    save_path = "/home/penguin/可视化result/"
    drawBBox(txt_path, img_path, save_path)
    print("All Done....")
