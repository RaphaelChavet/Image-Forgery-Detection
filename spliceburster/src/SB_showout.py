# This code is the demo of the SB (Splicebuster)
#    python SB_launcher.py input.png output.mat
#    python SB_showout.py input.png output.mat
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2018 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

from sys import argv
import numpy as np
import scipy.io as sio
from utility.utilityImage import imread2f
from utility.utilityImage import linear2uint8
from utility.utilityImage import resizeMapWithPadding
from utility.utilityImage import minmaxClip

imgfilename = argv[1]
outfilename = argv[2]

img = imread2f(imgfilename, channel = 3)
dat = sio.loadmat(outfilename)

map     = dat['map']
time    = dat['time'].flatten()
range0  = dat['range0'].flatten()
range1  = dat['range1'].flatten()
imgsize = dat['imgsize'].flatten()
print('time: %g' % time)

mapUint8 = linear2uint8(map)
mapUint8 = resizeMapWithPadding(mapUint8,range0,range1, imgsize)
[mapMin, mapMax] = minmaxClip(map, p = 0.02)
map[np.isnan(map)] = 0.0

try:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, clim=[0,1])
    plt.axis('off')
    plt.title('input \n image')
    plt.subplot(1,3,2)
    plt.imshow(map, clim=[mapMin, mapMax], cmap='jet')
    plt.axis('off')
    plt.title('result')
    plt.subplot(1,3,3)
    plt.imshow(255-mapUint8, clim=[0,255], cmap='gray')
    plt.xticks(list())
    plt.yticks(list())
    plt.title('result converted in uint8')
    plt.show()
except:
    print('warning: I cannot show the result');
