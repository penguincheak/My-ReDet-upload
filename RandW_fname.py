# import os
# import shutil
#
# # 想要移动文件所在的根目录
# rootdir = "/home/penguin/EDisk/gaofen-project/test/labelTxt"
# # 获取目录下文件名清单
# list = os.listdir(rootdir)
# f=open('/home/penguin/EDisk/gaofen-project/test/testset.txt','w')
# # print(files)
#
# # 移动图片到指定文件夹
# i = 0
# for item in list:  # 遍历该文件夹中的所有文件
#     item = item[0 : item.rfind('.', 1)] + '\n'
#     f.writelines(item)
#
# f.close()

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-11, 11, 100)
y = 1 / (1 + (np.e) ** (-x))

plt.figure(num=3, figsize=(8, 5))

plt.plot(x, y, color='red', linewidth=1.0, linestyle='--')

# plt.xlim(-1, 2)
# plt.ylim(-2, 3)
plt.xlabel('x')
plt.ylabel('y')

ax=plt.gca()
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))

ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

plt.show()