import json
import os
import os.path as osp
import numpy as np
import xmltodict

from dota_poly2rbox import get_best_begin_point_single


def parse_ann_info(objects):
    bboxes, labels = [], []
    # only one annotation
    if type(objects) != list:
        objects = [objects]
    for obj in objects:
        label = obj['possibleresult']['name']
        point = obj['points']['point']
        bbox = []
        count = 0
        for p in point:
            if(count == 4):
                break
            count += 1
            x1, x2 = p.split(',')
            bbox.append(float(x1))
            bbox.append(float(x2))
        bbox = tuple(bbox)
        bboxes.append(bbox)
        labels.append(label)
    return bboxes, labels

def ann_to_txt(ann):
    out_str = ''
    for bbox, label in zip(ann['bboxes'], ann['labels']):
        poly = np.array(bbox, dtype=np.float32)
        poly = get_best_begin_point_single(poly)
        str_line = '{} {} {} {} {} {} {} {} {} {}\n'.format(
            poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7], label, '0')
        out_str += str_line
    return out_str


def generate_txt_labels(root_path):
    img_path = osp.join(root_path, 'images')
    label_path = osp.join(root_path, 'labelXml')
    label_txt_path = osp.join(root_path, 'labelTxt')
    if not osp.exists(label_txt_path):
        os.mkdir(label_txt_path)

    img_names = [osp.splitext(img_name.strip())[0] for img_name in os.listdir(img_path)]
    for img_name in img_names:
        label = osp.join(label_path, img_name + '.xml')
        label_txt = osp.join(label_txt_path, img_name + '.txt')
        f_label = open(label)
        data_dict = xmltodict.parse(f_label.read())
        data_dict = data_dict['annotation']
        f_label.close()
        label_txt_str = ''
        # with annotations
        if data_dict['objects']:
            objects = data_dict['objects']['object']
            bboxes, labels = parse_ann_info(
                objects)
            ann = dict(
                bboxes=bboxes,
                labels=labels)
            label_txt_str = ann_to_txt(ann)
        with open(label_txt, 'w') as f_txt:
            f_txt.write(label_txt_str)


if __name__ == '__main__':
    generate_txt_labels('/home/penguin/EDisk/gaofen/train')
    print('done!')