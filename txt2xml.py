import os, sys
import glob
from PIL import Image
###将txt格式文件转换为xml格式文件
# 图像的 ground truth 的 txt 文件存放位置
#windows 系统路径用“\”,ubuntu 中用“/”
src_txt_dir = "/home/penguin/result_project/"
src_xml_dir = "/home/penguin/test/"
#glob 返回的文件名只包括当前目录里的文件名，不包括子文件夹里的文件。
#print(img_Lists)

txtlist = os.listdir(src_txt_dir)
for txt in txtlist:
    f = open(src_txt_dir + txt)
    filename = txt[0: txt.rfind('.', 1)]
    xml_file = open(src_xml_dir + filename + '.xml', 'w')

    #print(xml_file )输出为xml的路径+名字文件，处于可写入模式
    xml_file.write('<?xml version="1.0" encoding="utf-8"?>\n')
    xml_file.write('<annotation>\n')#通过write()函数向文件中写入多行
    xml_file.write('    <source>')
    xml_file.write('        <filename>' + filename + '.tif' + '</filename>\n')
    xml_file.write('        <origin>GF2/GF3</origin>\n')
    xml_file.write('    </source>\n')
    xml_file.write('    <research>\n')
    xml_file.write('        <version>1.0</version>\n')
    xml_file.write('        <provider> NJUST </provider>\n')
    xml_file.write('        <author> 我们也要去厦门 </author>\n')
    xml_file.write('        <pluginname>FAIR1M</pluginname>\n')
    xml_file.write('        <pluginclass>object detection</pluginclass>\n')
    xml_file.write('        <time>2021-09</time>\n')
    xml_file.write('    </research>\n')
    xml_file.write('    <objects>\n')

    line = f.readline()
    while(line):
        line = line[0: line.rfind('\n')]
        line = line.split(' ')
        cls_name = ''
        if len(line) < 10:
            line = f.readline()
            continue
        if len(line) == 10:
            cls_name = line[-1]
        else:
            for j in range(len(line) - 9):
                if j <(len(line) - 10):
                    cls_name += line[9 + j] + ' '
                else:
                    cls_name += line[9 + j]

        xml_file.write('        <object>\n')
        xml_file.write('            <coordinate>pixel</coordinate>\n')
        xml_file.write('            <type>rectangle</type>\n')
        xml_file.write('            <description>None</description>\n')
        xml_file.write('')
        xml_file.write('            <possibleresult>\n')
        xml_file.write('                <name>' + cls_name + '</name>\n')
        xml_file.write('                <probability>' + line[0] + '</probability>\n')
        xml_file.write('            </possibleresult>\n')

        xml_file.write('            <points>\n')
        xml_file.write('                <point>' + line[1] + ',' + line[2] + '</point>\n')
        xml_file.write('                <point>' + line[3] + ',' + line[4] + '</point>\n')
        xml_file.write('                <point>' + line[5] + ',' + line[6] + '</point>\n')
        xml_file.write('                <point>' + line[7] + ',' + line[8] + '</point>\n')
        xml_file.write('                <point>' + line[1] + ',' + line[2] + '</point>\n')
        xml_file.write('            </points>\n')
        xml_file.write('        </object>\n')
        line = f.readline()
    xml_file.write('    </objects>\n')
    xml_file.write('</annotation>')