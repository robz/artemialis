import matplotlib.pyplot as plt
import torch
import time

title = 'Documents/PerformanceMidi/plot_test-%s.png' % time.strftime('%m-%d-%Y-%H-%M-%S')
print(title)

plt.plot(torch.rand(10))
plt.savefig(title)
plt.close()
plt.plot(torch.rand(10))
title = 'Documents/PerformanceMidi/plot_test2-%s.png' % time.strftime('%m-%d-%Y-%H-%M-%S')
plt.savefig(title)
