import os
import re

pre_label_path = '/home/penguin/DataSet/NWPU VHR-10 dataset/NWPU VHR-10 dataset/ground truth/'
pro_label_path = '/home/penguin/DataSet/NWPU VHR-10 dataset/NWPU VHR-10 dataset/label/'

classes = ['airplane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
           'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'vehicle']

labels = os.listdir(pre_label_path)

i = 0

for label in labels:
    f = open(pre_label_path + label)
    pro_label = open(pro_label_path + label, 'a')
    line = f.readline()
    while(line):
        if line == '\n':
            break
        class_num = line[line.rfind(',') + 1: line.rfind('\n')]
        line = re.findall(r'[(](.*?)[)],',line)
        try:
            l1 = line[0].split(',')
            l2 = line[1].split(',')
            x1, y1 = l1[0].strip(), l1[1].strip()
            x2, y2 = l2[0].strip(), l2[1].strip()
        except:
            print(label)

        conference = x1 + ' ' + y1 + ' ' + x1 + ' ' + y2 + ' ' \
                     + x2 + ' ' + y2 + ' ' + x2 + ' ' + y1 + ' ' + classes[int(class_num) - 1] + ' ' + '0' + '\n'
        pro_label.write(conference)
        line = f.readline()
    print(i)
    i = i + 1
    f.close()
    pro_label.close()