import os
path_label = '/home/penguin/EDisk/gaofen/train/labelTxt-Origion'
labels = os.listdir(path_label)  #路径
# train_val = open('/home/penguin/Experiments/ReDet/data/gaofen-challenge/train/testset.txt','w')
path = '/home/penguin/EDisk/gaofen-project/labelTxt/'
for label in labels:
    f = open(path_label + '/' + label)  # 返回一个文件对象
    target_label = open(path + label, 'w')
    line = f.readline()  # 调用文件的 readline()方法
    while line:
        line = line.split(' ')
        str = ''
        if len(line) > 10:
            for j in range(len(line) - 9):
                if j <(len(line) - 10):
                    str += line[8 + j] + ' '
                else:
                    str += line[8 + j]
        else:
            str = line[8]

        Court = ['Baseball Field', 'Basketball Court', 'Football Field', 'Tennis Court']
        Road = ['Roundabout', 'Intersection', 'Bridge']
        if str in Court or str in Road:
            line = f.readline()
            continue

        if str in ('Small Car', 'Bus', 'Cargo Truck', 'Dump Truck', 'Van',
                        'Trailer', 'Tractor', 'Excavator', 'Truck Tractor', 'other-vehicle'):
            str = 'Vehicle'
        elif str in ('Passenger Ship', 'Motorboat', 'Fishing Boat', 'Tugboat',
                         'Engineering Ship', 'Liquid Cargo Ship', 'Dry Cargo Ship', 'Warship', 'other-ship'):
            str = 'Ship'
        elif str in ('Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'ARJ21','C919', 'A220',
                         'A321', 'A330', 'A350', 'other-airplane'):
            str = 'Airplane'

        _line = ''
        for j in range(0, 8):
            _line += line[j] + ' '
        _line += str + ' ' + line[-1]
        target_label.write(_line+'\n')
        line = f.readline()
    f.close()
    target_label.close()