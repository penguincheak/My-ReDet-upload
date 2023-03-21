import os
import math
import argparse
import os.path as osp

import numpy as np

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_single(coordinate):
    x1 = coordinate[0][0]
    y1 = coordinate[0][1]
    x2 = coordinate[1][0]
    y2 = coordinate[1][1]
    x3 = coordinate[2][0]
    y3 = coordinate[2][1]
    x4 = coordinate[3][0]
    y4 = coordinate[3][1]
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) + cal_line_length(combinate[i][1],
                                                                                           dst_coordinate[
                                                                                               1]) + cal_line_length(
            combinate[i][2], dst_coordinate[2]) + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return  combinate[force_flag]

def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],
                       poly[1][0], poly[1][1],
                       poly[2][0], poly[2][1],
                       poly[3][0], poly[3][1]
                       ]
    return outpoly

def get_best_begin_point_warp_single(coordinate):

    return TuplePoly2Poly(get_best_begin_point_single(coordinate))

def get_best_begin_point(coordinate_list):
    best_coordinate_list = map(get_best_begin_point_warp_single, coordinate_list)
    # import pdb
    # pdb.set_trace()
    best_coordinate_list = np.stack(list(best_coordinate_list))

    return best_coordinate_list

def angle_count(label_path, result_path):
    label_list = os.listdir(label_path)
    result_w = open(result_path, 'w')
    for label in label_list:
        print(label)
        file = open(label_path + label, 'r')
        line = file.readline()
        line = file.readline()
        line = file.readline()
        while(line):
            line = line.split(' ')
            line = line[:-2]
            line = list(map(float, line))
            i = 0
            bb = []
            while i < len(line):
                cord = [line[i], line[i + 1]]
                bb.append(cord)
                i += 2
            bb = np.array(bb)
            bbox = []
            bbox.append(bb)
            bbox = get_best_begin_point(bbox)
            bbox = np.array(bbox, dtype=np.float32)
            bbox = np.reshape(bbox, newshape=(-1, 2, 4), order='F')

            angle = np.arctan2(-(bbox[:, 0, 1] - bbox[:, 0, 0]), bbox[:, 1, 1] - bbox[:, 1, 0])

            angle = angle % (2 * np.pi)
            angle = angle / np.pi * 180
            result_w.write(str(angle[0]) + '\n')
            line = file.readline()
    result_w.close()


if __name__ == '__main__':
    label_txt = "/home/penguin/dota/hrsc2016/labelTxt/"
    result_path = "/home/penguin/dota/hrsc2016/angle_result.txt"
    angle_count(label_txt, result_path)