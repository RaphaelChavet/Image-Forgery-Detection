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

import numpy as np
from PIL import Image
from scipy.interpolate import interp1d

def imread2f(strFile, channel = 1, dtype = np.float32):
    img = Image.open(strFile)
    if (channel==3):
        img = img.convert('RGB')
        img = np.asarray(img).astype(dtype) / 256.0
    elif (channel==1):
        if img.mode == 'L':
            img = np.asarray(img).astype(dtype) / 256.0
        else:
            img = img.convert('RGB')
            img = np.asarray(img).astype(dtype)
            img = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2])/256.0
    else:
        img = np.asarray(img).astype(dtype) / 256.0

    return img

def img2grayf(img, dtype = np.float32):
    in_dtype = img.dtype
    img = img.astype(dtype)

    if in_dtype == np.uint8:
        img = img/256.0
    elif in_dtype == np.uint16:
        img = img/65536.0
    elif in_dtype == np.uint32:
        img = img/(2.0**32)
    elif in_dtype == np.uint64:
        img = img/(2.0**64)

    if (img.ndim == 3):
        if (img.shape[2] == 3):
            img = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])

    if img.ndim>2: img = np.mean(img,axis=2)

    return img

def resizeMapWithPadding(x, range0, range1, shapeOut):
    range0 = range0.flatten()
    range1 = range1.flatten()
    xv = np.arange(shapeOut[1])
    yv = np.arange(shapeOut[0])
    y = interp1d(range1, x    , axis=1, kind='nearest', fill_value='extrapolate', assume_sorted=True, bounds_error=False)
    y = interp1d(range0, y(xv), axis=0, kind='nearest', fill_value='extrapolate', assume_sorted=True, bounds_error=False)
    return y(yv).astype(x.dtype)

def linear2uint8(x):
    tM = np.nanmax(x)
    tm = np.nanmin(x)
    y = (255* (x - tm) /(tM-tm)).astype(np.uint8)
    return y
 
def minmaxClip(x,p = 0.05):
    x = x[np.isfinite(x)]
    x = np.sort(x.flatten())
    a = x[int((x.size)*   p )]
    b = x[int((x.size)*(1-p))]
    return a, b
