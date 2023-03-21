import os, sys
import glob
from PIL import Image
###将txt格式文件转换为xml格式文件
# 图像的 ground truth 的 txt 文件存放位置
#windows 系统路径用“\”,ubuntu 中用“/”
src_txt_dir = "/home/penguin/EDisk/gaofen/test/testset.txt"
src_xml_dir = "/home/penguin/test/"
#glob 返回的文件名只包括当前目录里的文件名，不包括子文件夹里的文件。
#print(img_Lists)
ccount = 0

testset = open(src_txt_dir)
line = testset.readline()
while(line):
    line = line[0: line.rfind('\n')]
    xml_name = line + '.xml'
    if os.path.exists(src_xml_dir + xml_name):
        line = testset.readline()
        continue
    else:
        ccount += 1
        print(xml_name)
        xml_file = open(src_xml_dir + xml_name, 'w')
        # print(xml_file )输出为xml的路径+名字文件，处于可写入模式
        xml_file.write('<?xml version="1.0" encoding="utf-8"?>\n')
        xml_file.write('<annotation>\n')  # 通过write()函数向文件中写入多行
        xml_file.write('    <source>')
        xml_file.write('        <filename>' + line + '.tif' + '</filename>\n')
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
        xml_file.write('    </objects>\n')
        xml_file.write('</annotation>')
    line = testset.readline()

print(ccount)