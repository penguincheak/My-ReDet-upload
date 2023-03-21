import json
import os
import os.path as osp
import numpy as np
import xmltodict

from dota_poly2rbox import get_best_begin_point_single


def parse_ann_info(objects):
    bboxes, labels, bboxes_ignore, labels_ignore = [], [], [], []
    # only one annotation
    if type(objects) != list:
        objects = [objects]
    for obj in objects:
        label = obj['name']
        bndbox = obj['bndbox']
        bbox = float(bndbox['xmin']), float(bndbox['ymin']), float(bndbox['xmax']), float(
            bndbox['ymax'])
        bboxes.append(bbox)
        labels.append(label)
    return bboxes, labels, bboxes_ignore, labels_ignore


def rbox2poly_single(rrect):
    """
    rrect:[xmin,ymin,xmax,ymax]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    xmin, ymin, xmax, ymax = rrect[:4]
    x0, x1, x2, x3 = xmin, xmax, xmax, xmin
    y0, y1, y2, y3 = ymin, ymin, ymax, ymax
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
    poly = get_best_begin_point_single(poly)
    return poly

def ann_to_txt(ann):
    out_str = ''
    for bbox, label in zip(ann['bboxes'], ann['labels']):
        poly = rbox2poly_single(bbox)
        str_line = '{} {} {} {} {} {} {} {} {} {}\n'.format(
            poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7], label, '0')
        out_str += str_line
    for bbox, label in zip(ann['bboxes_ignore'], ann['labels_ignore']):
        poly = rbox2poly_single(bbox)
        str_line = '{} {} {} {} {} {} {} {} {} {}\n'.format(
            poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], poly[6], poly[7], label, '1')
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
        if data_dict['object']:
            objects = data_dict['object']
            bboxes, labels, bboxes_ignore, labels_ignore = parse_ann_info(
                objects)
            ann = dict(
                bboxes=bboxes,
                labels=labels,
                bboxes_ignore=bboxes_ignore,
                labels_ignore=labels_ignore)
            label_txt_str = ann_to_txt(ann)
        with open(label_txt, 'w') as f_txt:
            f_txt.write(label_txt_str)


if __name__ == '__main__':
    generate_txt_labels('/home/penguin/EDisk/监督学习/train/')
    generate_txt_labels('/home/penguin/EDisk/监督学习/test/')
    print('done!')