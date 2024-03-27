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
from feat_spam.residue import getFiltersResidue
import feat_spam.mapping as spam_m
from scipy.signal import correlate2d

my_round = lambda x: np.sign(x) * np.floor(np.abs(x)+0.5)

def quantizerScalarEncoder(x, values):
    y = np.zeros(x.shape, dtype = np.int64)
    th = (values[1:]+values[:-1]) / 2
    for index in range(th.size):
        y += x>th[index]
    return y

def getParams(ordResid, symTranspose, q, T, ordCooc, mapper, strides):
    Wres, Fres, resTranspose = getFiltersResidue(ordResid)
    radius = (np.asarray(Wres.shape[0:2]) - 1) / 2

    n = 2*T + 1
    values = (float(q) * Fres / 256.0) * np.asarray(range(-T,T+1)).astype(np.float)

    radius = radius + (ordCooc - (ordCooc % 2)) / 2
    radius = radius.astype(int)
    
    if isinstance(mapper,dict):
        numFeat = mapper['num']
    elif mapper is 'SignSym':
        mapper = spam_m.getSignSymMapper(ordCooc, n)
        numFeat = mapper['num']
    elif mapper is 'Sign':
        mapper = spam_m.getSignMapper(ordCooc,n)
        numFeat = mapper['num']
    elif mapper is 'Idem':
        mapper = []
        numFeat = (n ** ordCooc)
    else:
        mapper = spam_m.getSignSymMapper(ordCooc, n)
        numFeat = mapper['num']

    if isinstance(strides, (list, tuple)):
        strides = strides[0:2]
    else:
        strides = [strides, strides]

    return {'Wres': Wres, 'resTranspose': resTranspose, 'uniformQuant': True, 'values': values, 'ordCooc': ordCooc, 'mapper': mapper,
            'strides': strides, 'numFeat': numFeat, 'radius': radius,
            'symTranspose': symTranspose}



def computeClip(X, params, normalize = True):

    radius  = params['radius']
    strides = params['strides']

    X = X[radius[0]:-radius[0], :]
    X = X[:, radius[1]:-radius[1]]

    range0 = range(0, X.shape[0] - strides[0] + 1, strides[0])
    range1 = range(0, X.shape[1] - strides[1] + 1, strides[1])
    Y = np.zeros([len(range0), len(range1)])
    for index0 in range(len(range0)):
        for index1 in range(len(range1)):
            pos0 = range0[index0]
            end0 = strides[0] + pos0
            pos1 = range1[index1]
            end1 = strides[1] + pos1
            Y[index0,index1] = np.sum(X[pos0:end0, pos1:end1])

    if normalize:
        Y = Y / (strides[0] * strides[1])

    range0 = range0 + radius[0] + (strides[0] -1.0) / 2.0
    range1 = range1 + radius[1] + (strides[1] -1.0) / 2.0

    return Y, range0, range1


def computeSpam(X, params, weights = list(),  normalize = True):

    ## Residue
    WresH = params['Wres']
    WresV = WresH.transpose([1,0,2,3])
    resH = correlate2d(X.squeeze(), WresH.squeeze(), mode='valid')
    resV = correlate2d(X.squeeze(), WresV.squeeze(), mode='valid')

    ## Quantization & Truncation
    values = params['values']
    if params['uniformQuant']:
        n = values.size
        T = (n - 1) / 2
        q = (values[1] - values[0])
        resHq = (np.clip(my_round(resH / q) + T, 0, n - 1)).astype(np.int64)
        resVq = (np.clip(my_round(resV / q) + T, 0, n - 1)).astype(np.int64)
    else:
        resHq = quantizerScalarEncoder(resH, values)
        resVq = quantizerScalarEncoder(resV, values)

    ## Coocorance
    ordCooc = params['ordCooc']
    n = (params['values']).size
    dim = int(ordCooc + 1 - (ordCooc % 2))
    indexL = int((dim - 1) / 2)

    shapeR = np.asarray(resHq.shape[:2]) - dim + 1
    resHh = np.zeros(shapeR, dtype = np.int)
    resVh = np.zeros(shapeR, dtype = np.int)
    resHv = np.zeros(shapeR, dtype = np.int)
    resVv = np.zeros(shapeR, dtype = np.int)

    for indexP in range(ordCooc):
        nn = (n ** indexP)
        resHh += resHq[indexL:(shapeR[0] + indexL), indexP:(shapeR[1] + indexP)] * nn
        resVh += resVq[indexL:(shapeR[0] + indexL), indexP:(shapeR[1] + indexP)] * nn
        resHv += resHq[indexP:(shapeR[0] + indexP), indexL:(shapeR[1] + indexL)] * nn
        resVv += resVq[indexP:(shapeR[0] + indexP), indexL:(shapeR[1] + indexL)] * nn

    ## Mappeing
    mapper = params['mapper']
    if len(mapper) > 0:
        resHh = mapper['table'][resHh].squeeze()
        resVh = mapper['table'][resVh].squeeze()
        resHv = mapper['table'][resHv].squeeze()
        resVv = mapper['table'][resVv].squeeze()

    ## Hist
    strides = params['strides']
    numFeat = params['numFeat']

    shapeR = resHh.shape
    range0 = np.arange(0, shapeR[0]-strides[0]+1, strides[0], dtype=np.uint16)
    range1 = np.arange(0, shapeR[1]-strides[1]+1, strides[1], dtype=np.uint16)
    rangeH = np.arange(0, numFeat+1, dtype = resHh.dtype) # range(0, numFeat+1)
    spamHh = np.zeros([range0.size, range1.size, numFeat], dtype = np.float32)
    spamVh = np.zeros([range0.size, range1.size, numFeat], dtype = np.float32)
    spamHv = np.zeros([range0.size, range1.size, numFeat], dtype = np.float32)
    spamVv = np.zeros([range0.size, range1.size, numFeat], dtype = np.float32)
    spamW  = np.zeros([range0.size, range1.size, ], dtype=np.float32)

    if len(weights) > 0:
        radius = params['radius']
        weights = weights[radius[0]:-radius[0], radius[1]:-radius[1]] ## clip weights
        for index0 in range(range0.size):
            for index1 in range(range1.size):
                pos0 = range0[index0]
                end0 = strides[0]+pos0
                pos1 = range1[index1]
                end1 = strides[1]+pos1
                weights_loc = weights[pos0:end0, :]
                weights_loc = weights_loc[:, pos1:end1]
                weights_loc = weights_loc.astype(np.float32)
                spamW[index0, index1] = np.sum(weights_loc)
                if spamW[index0, index1]>0:
                    spamHh[index0, index1, :], _ = np.histogram(resHh[pos0:end0, pos1:end1], rangeH, density=normalize, weights=weights_loc)
                    spamVh[index0, index1, :], _ = np.histogram(resVh[pos0:end0, pos1:end1], rangeH, density=normalize, weights=weights_loc)
                    spamHv[index0, index1, :], _ = np.histogram(resHv[pos0:end0, pos1:end1], rangeH, density=normalize, weights=weights_loc)
                    spamVv[index0, index1, :], _ = np.histogram(resVv[pos0:end0, pos1:end1], rangeH, density=normalize, weights=weights_loc)
    else:
        spamW[:,:] = strides[0]*strides[1]
        for index0 in range(range0.size):
            for index1 in range(range1.size):
                pos0 = range0[index0]
                end0 = strides[0] + pos0
                pos1 = range1[index1]
                end1 = strides[1] + pos1
                spamHh[index0, index1, :], _ = np.histogram(resHh[pos0:end0, pos1:end1], rangeH, density=normalize)
                spamVh[index0, index1, :], _ = np.histogram(resVh[pos0:end0, pos1:end1], rangeH, density=normalize)
                spamHv[index0, index1, :], _ = np.histogram(resHv[pos0:end0, pos1:end1], rangeH, density=normalize)
                spamVv[index0, index1, :], _ = np.histogram(resVv[pos0:end0, pos1:end1], rangeH, density=normalize)

    if normalize:
        spamW = spamW / (strides[0]*strides[1])

    if params['symTranspose']:
        spam = np.concatenate([spamVh+spamHv, spamHh+spamVv], 2)
    else:
        spam = np.concatenate([spamHh, spamVv, spamHv, spamVh], 2)

    ## Border
    radius = params['radius']
    range0 = range0 + radius[0] + (strides[0] - 1.0) / 2.0
    range1 = range1 + radius[1] + (strides[1] - 1.0) / 2.0

    return spam, spamW, range0, range1


def getSpam(X, params, ksize, weights = list(), paddingModality = 0):

    strides = params['strides']
    if isinstance(ksize, (list,tuple)):
        ksize = ksize[0:2]
    else:
        ksize = [ksize, ksize]

    ksize[0] = int(ksize[0] / strides[0])
    ksize[1] = int(ksize[1] / strides[1])

    spam, spamW, range0, range1 = computeSpam(X, params, weights = weights, normalize=True)

    filt = np.ones(ksize, dtype = spam.dtype)/(ksize[0]*ksize[1])
    numFeat = spam.shape[2]
    if paddingModality == 1:
        spamW = correlate2d(spamW, filt, mode='same', boundary='fill', fillvalue=0.0)
        for index in range(numFeat):
            spam[:, :, index] = correlate2d(spam[:,:,index], filt, mode='same', boundary='fill', fillvalue=0.0)
            spam[:, :, index] /= spamW
    elif paddingModality == 2:
        spamWnew = correlate2d(spamW, filt, mode='same', boundary='fill', fillvalue=0.0)
        for index in range(numFeat):
            spam[:, :, index] = correlate2d(spam[:,:,index], filt, mode='same', boundary='fill', fillvalue=0.0)
            spam[:, :, index] /= spamWnew
        spamW = correlate2d(spamW, filt, mode='same', boundary='symm')
    else:
        spamOut = np.zeros((spam.shape[0]-ksize[0]+1,spam.shape[1]-ksize[1]+1, spam.shape[2]), dtype = spam.dtype)
        spamW = correlate2d(spamW, filt, mode='valid')
        for index in range(numFeat):
            spamOut[:, :, index] = correlate2d(spam[:,:,index], filt, mode='valid')
            spamOut[:, :, index] /= np.maximum(spamW,1e-20)

        ind0f = int(np.floor((ksize[0] - 1.0) / 2.0))
        ind0c = int(np.ceil( (ksize[0] - 1.0) / 2.0))
        ind1f = int(np.floor((ksize[1] - 1.0) / 2.0))
        ind1c = int(np.ceil( (ksize[1] - 1.0) / 2.0))
        range0 = (range0[ind0f:-ind0c] + range0[ind0c:-ind0f])/2.0
        range1 = (range1[ind1f:-ind1c] + range1[ind1c:-ind1f])/2.0
        spam = spamOut

    return spam, spamW, range0, range1

