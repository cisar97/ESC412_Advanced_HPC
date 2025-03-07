#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sys import argv

data = np.log(np.fromfile(argv[1],dtype=np.single))
n = int(sqrt(data.shape[0]))
data.shape = (n,n)
fig, ax = plt.subplots()
ax.set_axis_off()
#print(np.amin(data),np.amax(data))
#ax.imshow(data, cmap='gnuplot2', vmin=0.6931472, vmax=7.636752)
ax.imshow(data, cmap='gnuplot2')
if len(argv) > 2:
    plt.savefig(argv[2], bbox_inches='tight', pad_inches=0)
else:
    plt.show()
