import os
import matplotlib.pyplot as plt

angle_path = "/home/penguin/dota/hrsc2016/angle_result.txt"

angle_file = open(angle_path, 'r')
line = angle_file.readline()
# 1）准备数据
angle = []
while(line):
    line = line[:line.rfind('\n')]
    line = float(line)
    angle.append(line)
    line = angle_file.readline()

# 2）创建画布
plt.figure(figsize=(30, 15), dpi=100)

# 3）绘制直方图
# 设置组距
distance = 10
# 计算组数
group_num = int((max(angle) - min(angle)) / distance)
# 绘制直方图
plt.hist(angle, bins=group_num)

# 修改x轴刻度显示
plt.xticks(range(int(min(angle)), int(max(angle)) + 1)[::10] ,fontsize=8)

# 添加网格显示
plt.grid(linestyle="--", alpha=0.5)

# 添加x, y轴描述信息
plt.xlabel("angles")
plt.ylabel("numbers")

# 4）显示图像
plt.show()