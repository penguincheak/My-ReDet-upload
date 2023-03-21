import os

path = '/home/penguin/Experiments/ReDet/data/dior/trainval/labelTxt/'
path_2 = '/home/penguin/Experiments/ReDet/data/dior/test/labelTxt/'
labels = os.listdir(path)

mmax = 0
mmin = 1000000

for label in labels:
    f = open(path + label, 'r')
    line = f.readline()
    line = f.readline()
    line = f.readline()
    while(line):
          line = line.split(' ')
          line = line[:8]
          nums = []
          for num in line:
              nums.append(float(num))
          h = max(nums[0::2]) - min(nums[0::2])
          w = max(nums[1::2]) - min(nums[1::2])
          if mmax < h * w:
              mmax = h * w
          if mmin > h * w:
              mmin = h * w
          line = f.readline()

labels = os.listdir(path_2)
for label in labels:
    f = open(path_2 + label, 'r')
    line = f.readline()
    line = f.readline()
    line = f.readline()
    while(line):
          line = line.split(' ')
          line = line[:8]
          nums = []
          for num in line:
              nums.append(float(num))
          h = max(nums[0::2]) - min(nums[0::2])
          w = max(nums[1::2]) - min(nums[1::2])
          if mmax < h * w:
              mmax = h * w
          if mmin > h * w:
              mmin = h * w
          line = f.readline()

print('mmax:' + str(mmax))
print('mmin:' + str(mmin))
