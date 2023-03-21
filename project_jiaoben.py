import os, shutil

txt_path = '/home/penguin/result_project/' # 改成result_project的位置
image_project = '/home/penguin/EDisk/images_project/' # 改成结果保存位置
image_path = '/home/penguin/EDisk/gaofen_project/test/images/' # 改成test文件夹的位置

i = 0
thresh_dir = ['大于0.75', '0.25-0.75', '小于0.25', '小于0.05']

for file in os.listdir(txt_path):
    max_Iou = {}
    max_Iou['Ship'] = 0
    max_Iou['Airplane'] = 0
    max_Iou['Vehicle'] = 0
    f = open(txt_path + file)
    line = f.readline()
    while(line):
        split_line = line.split(' ')
        iou = float(split_line[0])
        cls = (split_line[-1])[0: split_line[-1].rfind('/n')]
        if iou > max_Iou[cls]:
            max_Iou[cls] = iou

        line = f.readline()
    if max_Iou['Ship'] > 0:
        image_name = file[0: file.rfind('.')] + '.tif'
        if max_Iou['Ship'] >= 0.75:
            shutil.copy(image_path + image_name,
                        image_project +'/' +thresh_dir[0] +'/' + 'Ship/' + str(max_Iou['Ship']) + "_" + image_name)
        elif max_Iou['Ship'] < 0.75 and max_Iou['Ship'] >= 0.25:
            shutil.copy(image_path + image_name,
                        image_project + '/' + thresh_dir[1] + '/' + 'Ship/' + str(max_Iou['Ship']) + "_" + image_name)
        elif max_Iou['Ship'] < 0.25 and max_Iou['Ship'] >= 0.05:
            shutil.copy(image_path + image_name,
                        image_project + '/' + thresh_dir[2] + '/' + 'Ship/' + str(max_Iou['Ship']) + "_" + image_name)
        elif max_Iou['Ship'] < 0.05:
            shutil.copy(image_path + image_name,
                        image_project + '/' + thresh_dir[3] + '/' + 'Ship/' + str(max_Iou['Ship']) + "_" + image_name)
    if max_Iou['Airplane'] > 0:
        image_name = file[0: file.rfind('.')] + '.tif'
        if max_Iou['Airplane'] >= 0.75:
            shutil.copy(image_path + image_name,
                        image_project +'/' +thresh_dir[0] +'/' + 'Airplane/' + str(max_Iou['Airplane']) + "_" + image_name)
        elif max_Iou['Airplane'] < 0.75 and max_Iou['Airplane'] >= 0.25:
            shutil.copy(image_path + image_name,
                        image_project + '/' + thresh_dir[1] + '/' + 'Airplane/' + str(max_Iou['Airplane']) + "_" + image_name)
        elif max_Iou['Airplane'] < 0.25 and max_Iou['Airplane'] >= 0.05:
            shutil.copy(image_path + image_name,
                        image_project + '/' + thresh_dir[2] + '/' + 'Airplane/' + str(max_Iou['Airplane']) + "_" + image_name)
        elif max_Iou['Airplane'] < 0.05:
            shutil.copy(image_path + image_name,
                        image_project + '/' + thresh_dir[3] + '/' + 'Airplane/' + str(
                            max_Iou['Airplane']) + "_" + image_name)
    if max_Iou['Vehicle'] > 0:
        image_name = file[0: file.rfind('.')] + '.tif'
        if max_Iou['Vehicle'] >= 0.75:
            shutil.copy(image_path + image_name,
                        image_project +'/' +thresh_dir[0] +'/' + 'Vehicle/' + str(max_Iou['Vehicle']) + "_" + image_name)
        elif max_Iou['Vehicle'] < 0.75 and max_Iou['Vehicle'] >= 0.25:
            shutil.copy(image_path + image_name,
                        image_project + '/' + thresh_dir[1] + '/' + 'Vehicle/' + str(max_Iou['Vehicle']) + "_" + image_name)
        elif max_Iou['Vehicle'] < 0.25 and max_Iou['Vehicle'] >= 0.05:
            shutil.copy(image_path + image_name,
                        image_project + '/' + thresh_dir[2] + '/' + 'Vehicle/' + str(max_Iou['Vehicle']) + "_" + image_name)
        elif max_Iou['Vehicle'] < 0.05:
            shutil.copy(image_path + image_name,
                        image_project + '/' + thresh_dir[3] + '/' + 'Vehicle/' + str(
                            max_Iou['Vehicle']) + "_" + image_name)
    i += 1
    print(i)
