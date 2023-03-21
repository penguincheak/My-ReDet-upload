import utils as util
import os
import ImgSplit_multi_process_gaofen
import SplitOnlyImage_multi_process_gaofen
import shutil
from multiprocessing import Pool
from GaoFen2COCO import GaoFen2COCOTest, GaoFen2COCOTrain
import argparse

# wordname_37 = ['Passenger Ship', 'Motorboat', 'Fishing Boat', 'Tugboat',
#                'Engineering Ship', 'Liquid Cargo Ship',
#                'Dry Cargo Ship', 'Warship', 'Small Car', 'Van', 'Dump Truck',
#                'Cargo Truck', 'Intersection', 'Truck Tractor',
#                'Bus', 'Tennis Court', 'Trailer', 'Excavator',
#                'A220', 'Football Field', 'Boeing737', 'Baseball Field', 'A321',
#                'Boeing787', 'Basketball Court', 'Boeing747', 'A330', 'Boeing777',
#                'Tractor', 'Bridge', 'A350', 'C919', 'ARJ21', 'Roundabout']

wordname_37 = ['Vehicle', 'Ship', 'Airplane']

def parse_args():
    parser = argparse.ArgumentParser(description='prepare gaofen')
    parser.add_argument('--srcpath', default='/home/penguin/EDisk/gaofen_project/')
    parser.add_argument('--dstpath', default='/home/penguin/EDisk/gaofen_project/',
                        help='prepare data')
    args = parser.parse_args()

    return args

def single_copy(src_dst_tuple):
    shutil.copyfile(*src_dst_tuple)
def filecopy(srcpath, dstpath, num_process=16):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(single_copy, name_pairs)

def singel_move(src_dst_tuple):
    shutil.move(*src_dst_tuple)

def filemove(srcpath, dstpath, num_process=16):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(filemove, name_pairs)

def getnamelist(srcpath, dstfile):
    filelist = util.GetFileFromThisRootDir(srcpath)
    with open(dstfile, 'w') as f_out:
        for file in filelist:
            basename = util.mybasename(file)
            f_out.write(basename + '\n')

def prepare(srcpath, dstpath):
    """
    :param srcpath: train, val, test
          trainval --> trainval1000, test --> test1000
    :return:
    """
    if not os.path.exists(os.path.join(dstpath, 'test1000')):
        os.makedirs(os.path.join(dstpath, 'test1000'))
    # if not os.path.exists(os.path.join(dstpath, 'test1000_ms')):
    #     os.makedirs(os.path.join(dstpath, 'test1000_ms'))
    if not os.path.exists(os.path.join(dstpath, 'trainval1000')):
        os.makedirs(os.path.join(dstpath, 'trainval1000'))
    # if not os.path.exists(os.path.join(dstpath, 'trainval1000_ms')):
    #     os.makedirs(os.path.join(dstpath, 'trainval1000_ms'))

    split_train = ImgSplit_multi_process_gaofen.splitbase(os.path.join(srcpath, 'train'),
                       os.path.join(dstpath, 'trainval1000'),
                      gap=200,
                      subsize=1000,
                      ext='.tif',
                      num_process=32
                      )
    split_train.splitdata(1)

    # split_train_ms = ImgSplit_multi_process_gaofen.splitbase(os.path.join(srcpath, 'train'),
    #                     os.path.join(dstpath, 'trainval1000_ms'),
    #                     gap=500,
    #                     subsize=1000,
    #                     ext='.tif',
    #                     num_process=32)
    # split_train_ms.splitdata(0.5)
    # split_train_ms.splitdata(1.5)

    split_test = SplitOnlyImage_multi_process_gaofen.splitbase(os.path.join(srcpath, 'test', 'images'),
                       os.path.join(dstpath, 'test1000', 'images'),
                      gap=200,
                      subsize=1000,
                      ext='.tif',
                      num_process=32
                      )
    split_test.splitdata(1)

    # split_test_ms = SplitOnlyImage_multi_process_gaofen.splitbase(os.path.join(srcpath, 'test', 'images'),
    #                    os.path.join(dstpath, 'test1000_ms', 'images'),
    #                   gap=500,
    #                   subsize=1000,
    #                   ext='.tif',
    #                   num_process=32
    #                   )
    # split_test_ms.splitdata(0.5)
    # split_test_ms.splitdata(1.5)

    GaoFen2COCOTrain(os.path.join(dstpath, 'trainval1000'), os.path.join(dstpath, 'trainval1000', 'gaofen_trainval1000.json'), wordname_37)
    # GaoFen2COCOTrain(os.path.join(dstpath, 'trainval1000_ms'), os.path.join(dstpath, 'trainval1000_ms', 'gaofen_trainval1000_ms.json'), wordname_37)

    GaoFen2COCOTest(os.path.join(dstpath, 'test1000'), os.path.join(dstpath, 'test1000', 'gaofen_test1000.json'), wordname_37)
    # GaoFen2COCOTest(os.path.join(dstpath, 'test1000_ms'), os.path.join(dstpath, 'test1000_ms', 'gaofen_test1000_ms.json'), wordname_37)
if __name__ == '__main__':
    args = parse_args()
    srcpath = args.srcpath
    dstpath = args.dstpath
    prepare(srcpath, dstpath)