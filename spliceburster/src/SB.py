# This is the code of the SB (Splicebuster)
#    algorithm described in "Splicebuster: a new blind image splicing detector",
#    written by  D. Cozzolino, G. Poggi and L. Verdoliva,
#    IEEE International Workshop on Information Forensics and Security, 2015.
#    Please, refer to this paper for any further information and
#     cite it every time you use the code.
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

import numpy as np
import skimage.morphology as ski
from feat_spam.spam_np import getParams as getSpamParams
from feat_spam.spam_np import getSpam
from utility.utilityImage import img2grayf
from utility.gaussianMixture import gm
from time import time

defaultSpamParam = getSpamParams(ordResid=3, symTranspose=True, q=2, T=1, ordCooc=4, mapper='SignSym', strides=8)
smallSpamParam   = getSpamParams(ordResid=3, symTranspose=True, q=2, T=1, ordCooc=4, mapper='SignSym', strides=4)

def getSatMap(img, th_Black, th_White):
    if len(img.shape) > 2:
        sat_mask0 = ski.binary_opening(img[:,:,0] < th_Black, ski.disk(3))
        sat_mask1 = ski.binary_opening(img[:,:,0] > th_White, ski.disk(2))
        for index in range(1, img.shape[2]):
            sat_mask0 = np.logical_or(sat_mask0, ski.binary_opening(img[:, :, index] < th_Black, ski.disk(3)))
            sat_mask1 = np.logical_or(sat_mask1, ski.binary_opening(img[:, :, index] > th_White, ski.disk(2)))
    else:
        sat_mask0 = ski.binary_opening(img<th_Black,ski.disk(3))
        sat_mask1 = ski.binary_opening(img>th_White,ski.disk(2))

    return np.logical_or(sat_mask0,sat_mask1)

def faetReduce(feat_list, inds, whiteningFlag = False):
    cov_mtx = np.cov(feat_list, rowvar = False, bias = True)
    w, v = np.linalg.eigh(cov_mtx)
    w = w[::-1]
    v = v[:,::-1]
    v = v[:, inds]
    if whiteningFlag:
        v = v / np.sqrt(w[inds])

    return v

def SBgu(img, paramSpam = defaultSpamParam, ksize = 128, flagDropSaturated = 2, satutationProb = 0.85, extFeat = range(25), seed = 0, maxIter = 100, replicates = 30, outliersNlogl = 42, paddingModality = 0, flagShow = False):
    if flagDropSaturated > 0:
        if img.dtype== np.uint8:
            weights = np.logical_not(getSatMap(img, 6, 252)).astype(np.float32)
        else:
            weights = np.logical_not(getSatMap(img, 6.0 / 256, 252.0 / 256)).astype(np.float32)
        weights = ski.binary_erosion(weights, np.ones((2*paramSpam['radius'][0]+1, 2*paramSpam['radius'][1]+1), dtype = np.bool))
    else:
        weights = list()

    timestamp = time()
    img = img2grayf(img)
    spam, weights, range0, range1 = getSpam(img, paramSpam, ksize, weights = weights, paddingModality = paddingModality)
    #print('FE: %g' % (time() - timestamp))

    weights = (weights >= satutationProb)
    spam = np.sqrt(spam)

    shape_spam = spam.shape

    list_spam  = spam.reshape([shape_spam[0]*shape_spam[1],shape_spam[2]])
    list_valid = list_spam[weights.flatten(),:]

    if list_valid.shape[0] == 0:
        other = dict()
        other['Sigma'] = 0
        other['mu'] = 0
        other['L'] = 0
        other['outliersNlogl'] = outliersNlogl
        other['outliersProb'] = 1
        return [], [], [], other
    
    L = faetReduce(list_valid, extFeat, True)
    list_spam = np.matmul(list_spam, L)
    list_valid = list_spam[weights.flatten(), :]

    timestamp = time()
    randomState = np.random.RandomState(seed)
    gm_data = gm(shape_spam[2], [0,], [2,], outliersProb = 0.01, outliersNlogl = outliersNlogl, dtype=list_valid.dtype)
    gm_data.setRandomParams(list_valid, regularizer = -1.0, randomState = randomState)
    avrLogl, _, _ = gm_data.EM(list_valid, maxIter = maxIter, regularizer = -1.0)
    #print('GT0: %g' % (time() - timestamp))

    if flagShow:
        #print(L.shape)
        Lu,Ls,Lv = np.linalg.svd(L, full_matrices=False)
        Ls = np.linalg.inv(np.matmul(Lv, np.diag(Ls)))
        SS = np.matmul(np.matmul(Ls, gm_data.listSigma[0]), np.transpose(Ls,[1,0]))
        SS = np.linalg.eigvalsh(SS)
        P1 = np.sum(SS)
        SS = SS/np.sum(SS)

        S1 = -np.sum(np.log2(SS) * SS)
        SS = np.linalg.eigvalsh(gm_data.listSigma[0])
        P2 = np.sum(SS)
        SS = SS / np.sum(SS)
        S2 = -np.sum(np.log2(SS) * SS)

        import matplotlib.pyplot as plt
        _, mahal = gm_data.getNlogl(list_spam)
        mahal = mahal.reshape([shape_spam[0], shape_spam[1], ])
        if flagDropSaturated > 1: mahal[np.logical_not(weights)] = np.nan
        plt.subplot(4,5,1)
        plt.imshow(mahal)
        plt.title('%f %f %f %f %f' %(avrLogl, S1, P1, S2, P2))

    for index in range(1, replicates):
        timestamp = time()
        gm_data_1 = gm(shape_spam[2], [0,], [2,], outliersProb = 0.01, outliersNlogl = outliersNlogl, dtype = list_valid.dtype)
        gm_data_1.setRandomParams(list_valid, regularizer = -1.0, randomState = randomState)

        avrLogl_1, _, _ = gm_data_1.EM(list_valid, maxIter = maxIter, regularizer = -1.0)
        if (avrLogl_1>avrLogl):
            gm_data  = gm_data_1
            avrLogl  = avrLogl_1

        if flagShow and (index<20):
            _, mahal = gm_data_1.getNlogl(list_spam)
            mahal = mahal.reshape([shape_spam[0], shape_spam[1], ])
            if flagDropSaturated > 1: mahal[np.logical_not(weights)] = np.nan
            plt.subplot(4, 5, index+1)
            plt.imshow(mahal)
            plt.title(avrLogl_1)
        #print('GT%d: %g' % (index, time() - timestamp))

    _, mahal = gm_data.getNlogl(list_spam)

    mahal = mahal.reshape([shape_spam[0],shape_spam[1],])

    if flagDropSaturated > 1:
        mahal[np.logical_not(weights)] = np.nan

    other = dict()
    other['Sigma'] = gm_data.listSigma[0]
    other['mu'] = gm_data.mu
    other['L'] = L
    other['weights'] = weights
    other['outliersNlogl'] = outliersNlogl
    other['outliersProb'] = gm_data.outliersProb

    return mahal, range0, range1, other

def SBsup(img, msk, paramSpam = defaultSpamParam, ksize = 128,  extFeat = range(25), paddingModality = 0):

    img = img2grayf(img)
    spam, weights, range0, range1 = getSpam(img, paramSpam, ksize, paddingModality = paddingModality)
    spam = np.sqrt(spam)

    msk = msk[range0.astype(np.int), :]
    msk = msk[:, range1.astype(np.int)]

    shape_spam = spam.shape
    list_spam  = spam.reshape([shape_spam[0]*shape_spam[1],shape_spam[2]])
    list_valid = list_spam[msk.flatten(),:]

    L = faetReduce(list_valid, extFeat, True)
    list_spam = np.matmul(list_spam, L)
    list_valid = list_spam[msk.flatten(), :]

    gm_data = gm(shape_spam[2], [0, ], [2, ], dtype = list_valid.dtype)
    post = np.ones([list_valid.shape[0],1], dtype = list_valid.dtype)
    gm_data.maximizationParam(list_valid, post, regularizer = -1.0)
    _, mahal = gm_data.getNlogl(list_spam)

    mahal = mahal.reshape([shape_spam[0],shape_spam[1],])

    other = dict()
    other['Sigma'] = gm_data.listSigma[0]
    other['mu'] = gm_data.mu
    other['L'] = L


    return mahal, range0, range1, other



from utility.utilityImage import imread2f

def SB_main(imgfilename):
    img = imread2f(imgfilename, channel =  1)
    imgsize = img.shape
    
    if imgsize[0]*imgsize[1]>20000:
         mapp, range0, range1, other = SBgu(img, paramSpam = defaultSpamParam, ksize = 128, flagDropSaturated=2, paddingModality=0)
    else:
         mapp, range0, range1, other = SBgu(img, paramSpam =   smallSpamParam, ksize =  64, flagDropSaturated=2, paddingModality=0)
    
    return mapp, range0, range1, imgsize, other
     
