import os
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

path = '/home/penguin/Experiments/ReDet/loss_and_acc.log'

f = open(path)

line = f.readline()
line = f.readline()
line = f.readline()
all_acc = []
all_loss = []
while(line):
    line = line.split(',')
    if(len(line) == 16):
        acc = line[13][line[13].rfind(':') + 2:]
        loss = line[15][line[15].rfind(':') + 2: line[15].rfind('\n')]
    elif(len(line) == 15):
        acc = line[12][line[12].rfind(':') + 2:]
        loss = line[14][line[14].rfind(':') + 2: line[14].rfind('\n')]

    all_acc.append(float(acc))
    all_loss.append(float(loss))

    line = f.readline()

x_axis = [i * 50 for i in range(len(all_acc))]

plt.figure()
pl.plot(x_axis, all_loss, 'r-', label=u'ReDet Loss', linewidth=0.3 * np.pi)
pl.legend()
pl.xlabel(u'iters')
pl.ylabel(u'loss')
plt.title('Loss Curve')
plt.savefig('loss.png')

plt.figure()
pl.plot(x_axis, all_acc, 'b-', label=u'ReDet Accuracy', linewidth=0.3 * np.pi)
pl.legend()
pl.xlabel(u'iters')
pl.ylabel(u'acc')
plt.title('Training Accuracy')
plt.savefig('acc.png')