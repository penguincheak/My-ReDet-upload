import os

root = '/home/penguin/EDisk/gaofen_project/test/labelTxt/'
w_root = '/home/penguin/EDisk/gaofen_project/test/test.txt'

for file in os.listdir(root):
    w = open(w_root, 'a')
    w.write(file[0:file.rfind('.txt')] + '\n')