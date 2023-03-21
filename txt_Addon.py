# 读写入
import os
import shutil
src_method_1_txt = "/home/penguin/Experiments/ReDet/work_dirs/ReDet_re50_refpn_3x_hrsc2016/Task1_results/"
src_method_2_txt = "/home/penguin/Experiments/ReDet/work_dirs/ReDet_re50_ressp_3x_hrsc2016/Task1_results/"

src_results_txt = "/home/penguin/Experiments/ReDet/work_dirs/ReDet_re50_refpn_3x_hrsc2016/Task1_results_res/"

if not os.path.exists(src_results_txt):
    os.mkdir(src_results_txt)

txt_list = os.listdir(src_method_1_txt)
for txt in txt_list:
    shutil.copy(src_method_1_txt + txt, src_results_txt + txt)
    f1 = open(src_results_txt + txt, 'a')
    f2 = open(src_method_2_txt + txt)
    f1.write('\n')
    line = f2.readline()
    while(line):
        f1.write(line)
        line = f2.readline()