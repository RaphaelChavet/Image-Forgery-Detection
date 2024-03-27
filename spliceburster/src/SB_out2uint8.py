# This code converts the SB output in a PNG image
#    
#    python SB_out2uint8.py output.mat output.png
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
from utility.utilityImage import linear2uint8
from utility.utilityImage import resizeMapWithPadding
from PIL import Image

matfilename = argv[1]
outfilename = argv[2]

dat = sio.loadmat(matfilename)
map     = dat['map']
time    = dat['time'].flatten()
range0  = dat['range0'].flatten()
range1  = dat['range1'].flatten()
imgsize = dat['imgsize'].flatten()

mapUint8 = linear2uint8(map)
mapUint8 = resizeMapWithPadding(mapUint8,range0,range1, imgsize)

Image.fromarray(255-mapUint8).save(outfilename)


