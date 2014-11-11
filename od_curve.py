#! /usr/bin/env python
# -*- coding: utf_8 -*-

from __future__ import print_function, division
import os
import numpy as np
import img_processor as imgp          # numpy image processing module
from matplotlib   import pyplot as plt
import math as m

path = os.curdir
files = [f for f in os.listdir(path) if 'OD' in f]
dc = [plt.imread(f).astype(np.float32) for f in os.listdir(path) if 'darkcount' in f]
print("darkcount image:\n", dc)
od = [0.0, 0.08, 0.14, 0.24, 0.34, 0.44, 0.54, 1.04, 1.54, 2.04]
pics = (plt.imread(f).astype(np.float32) for f in files)
normgen = (pic - dc for pic in pics)
meangen = (np.average(pic) for pic in normgen)


calc = True
if calc:
  mean = list(meangen)
  np.save('od', mean)
else:
  mean = np.load('od.npy')

#plot:

y = [m.log(n,10) for n in mean]
data = (od, y)
coord = zip(od, y)
m = coord[-1][1] - coord[-3][1]
yf = [m*o + 5 for o in od]

plt.plot(od, yf, *data)
plt.savefig('od.pdf')
plt.show()
