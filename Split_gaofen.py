from shutil import copy

classes = ['Airplane', 'Court', 'Road', 'Ship', 'Vehicle']
threshold = [0., 0.07, 0.07, 0.1, 0.13]

_img_root = r'/home/penguin/test/'
_label_root = r'/home/penguin/test/'
_img_save = r'/home/penguin/test_result/'
_label_save = r'/home/penguin/test_result/'

ind = 0
for cls in classes:
    label_root = _label_root + cls + '.txt'
    img_root = _img_root + cls + '/'
    img_save = _img_save + cls +'/'
    label_save = _label_save + cls + '.txt'
    file_r = open(label_root, 'r')
    file_w = open(label_save, 'w')
    line = file_r.readline()
    while(line):
        if line == '\n':
            line = file_r.readline()
            continue
        split_line = line[line.rfind('_', 1) + 1: -1]
        img_name = line[0: line.rfind('_', 1)] + '.jpg'
        split_line = split_line.split(' ')
        confidence = float(split_line[1])
        if confidence >= threshold[ind]:
            file_w.write(line + '\n')
            copy(img_root + img_name, img_save + img_name)
            print(img_name)
            line = file_r.readline()
        else:
            break

    ind = ind + 1
