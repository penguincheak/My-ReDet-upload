import os
import shutil

# 想要移动文件所在的根目录
rootdir = "/home/penguin/DataSet/NWPU VHR-10 dataset/NWPU VHR-10 dataset/positive image set/"
# 获取目录下文件名清单
list = os.listdir(rootdir)
# print(files)

# 移动图片到指定文件夹
i = 0
for item in list:  # 遍历该文件夹中的所有文件
    if i < 434:
        full_path = os.path.join(rootdir, item)  # 将文件目录与文件名连接起来，形成原来完整路径
        des_path = "/home/penguin/DataSet/NWPU VHR-10 dataset/NWPU VHR-10 dataset/train/images/"  # 目标路径
        shutil.copy(full_path, des_path + item)  # 移动文件到目标路径

        print(full_path)
        print(des_path)

        filename = item[0 : item.rfind('.', 1)]
        filename = filename + '.txt'

        full_path = "/home/penguin/DataSet/NWPU VHR-10 dataset/NWPU VHR-10 dataset/label/" + filename
        des_path = "/home/penguin/DataSet/NWPU VHR-10 dataset/NWPU VHR-10 dataset/train/labelTxt/"
        shutil.copy(full_path, des_path + filename)  # 移动文件到目标路径

        print(full_path)
        print(des_path)

    else:
        full_path = os.path.join(rootdir, item)  # 将文件目录与文件名连接起来，形成原来完整路径
        des_path = "/home/penguin/DataSet/NWPU VHR-10 dataset/NWPU VHR-10 dataset/test/images/"  # 目标路径
        shutil.copy(full_path, des_path + item)  # 移动文件到目标路径

        print(full_path)
        print(des_path)

        filename = item[0: item.rfind('.', 1)]
        filename = filename + '.txt'

        full_path = "/home/penguin/DataSet/NWPU VHR-10 dataset/NWPU VHR-10 dataset/label/" + filename
        des_path = "/home/penguin/DataSet/NWPU VHR-10 dataset/NWPU VHR-10 dataset/test/labelTxt/"
        shutil.copy(full_path, des_path + filename)  # 移动文件到目标路径

        print(full_path)
        print(des_path)

    i += 1