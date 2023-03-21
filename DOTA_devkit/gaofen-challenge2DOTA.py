import dota_utils as util
import os
import cv2
import json
from PIL import Image

L1_names = ['other', 'A220', 'Boeing737', 'A321', 'Boeing787', 'Boeing747', 'A330', 'Boeing777', 'A350', 'ARJ21']
# L1_names = ['target']
# TODO: finish them
L2_names = []
L3_names = []


def GaoFen2COCOTrain(srcpath, destfile, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'label_txt')

    data_dict = {}
    info = {'contributor': 'Chongyu Sun',
            'data_created': '2021',
            'description': 'This is the L1 of gaofen',
            'version': '1.0',
            'year': 2021}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        # with open(train_set_file, 'r') as f_in:
        #     lines = f_in.readlines()
        #     filenames = [os.path.join(labelparent, x.strip()) + '.txt' for x in lines]

        for file in filenames:
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.tif')

            img = Image.open(imagepath)
            height = img.height
            width = img.width

            print('height: ', height)
            print('width: ', width)

            single_image = {}
            single_image['file_name'] = basename + '.tif'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            for obj in objects:
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = cls_names.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)


def GaoFen2COCOTest(srcpath, destfile, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    # labelparent = os.path.join(srcpath, 'labelTxt')
    data_dict = {}
    info = {'contributor': 'Chongyu Sun',
            'data_created': '2019',
            'description': 'This is gaofen-challengee.',
            'version': '1.0',
            'year': 2021}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(imageparent)
        # with open(test_set_file, 'r') as f_in:
        #     lines = f_in.readlines()
        #     filenames = [os.path.join(imageparent, x.strip()) + '.bmp' for x in lines]

        for file in filenames:
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.tif')
            # img = cv2.imread(imagepath)
            img = Image.open(imagepath)
            # height, width, c = img.shape
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = basename + '.tif'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
        json.dump(data_dict, f_out)


if __name__ == '__main__':
    GaoFen2COCOTrain(r'data/gaofen-det/train',
                   r'data/gaofen-det/train/GaoFen_train.json',
                   L1_names)
    GaoFen2COCOTest(r'data/gaofen-det/test',
                  r'data/gaofen-det/test/GaoFen_test.json',
                  L1_names)
