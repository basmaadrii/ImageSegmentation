import numpy as np
import math
import os
import scipy.io as sio
import matplotlib.pyplot as plt


def Fmeasure(segmentation, gtSegmentation):

    k = np.max(gtSegmentation)                  #no of classes in GT
    r = int(np.max(segmentation))                    #no of clusters

    ti = np.zeros([k,])                         #no of points in class i
    ni = np.zeros([r,])                         #no of points in cluster i

    for i in range(0, r):
        ind = np.nonzero(segmentation == (i + 1))
        ni[i] = np.array(ind).shape[1]

    for i in range(0, k):
        ind = np.nonzero(gtSegmentation == (i + 1))
        ti[i] = np.array(ind).shape[1]

    prec = np.zeros([r,])
    rec = np.zeros([r,])
    F = 0
    for i in range(0, r):
        #finding values of cluster i
        ind = np.nonzero(segmentation == (i + 1))

        #getting gt in same indices of cluster i
        gt = np.zeros(int(ni[i]),)
        for j in range(0, int(ni[i])):
            gt[j] = gtSegmentation[ind[0][j], ind[1][j]]

        #counting classes in cluster i
        count = np.zeros([k,])
        for j in range(0, k):
            ind = np.nonzero(gt == (j + 1))
            count[j] = np.array(ind).shape[1]

        #calculating prec, rec, and F
        prec[i] = np.max(count / ni[i])
        if i > k:
            rec[i] = 0
        else:
            rec[i] = np.max(count / ti[i])

        F += (2 * prec[i] * rec[i]) / (prec[i] + rec[i])

    return F


def ConditionalEntropy(segmentation, gtSegmentation):
    k = np.max(gtSegmentation)  # no of classes in GT
    r = int(np.max(segmentation))  # no of clusters

    ti = np.zeros([k, ])  # no of points in class i
    ni = np.zeros([r, ])  # no of points in cluster i

    for i in range(0, r):
        ind = np.nonzero(segmentation == (i + 1))
        ni[i] = np.array(ind).shape[1]

    for i in range(0, k):
        ind = np.nonzero(gtSegmentation == (i + 1))
        ti[i] = np.array(ind).shape[1]

    Hi = np.zeros(r,)
    H = 0
    for i in range(0, r):
        #finding values of cluster i
        ind = np.nonzero(segmentation == (i + 1))

        #getting gt in same indices of cluster i
        gt = np.zeros(int(ni[i]),)
        for j in range(0, int(ni[i])):
            gt[j] = gtSegmentation[ind[0][j], ind[1][j]]

        #counting classes in cluster i
        count = np.zeros([k,])
        for j in range(0, k):
            ind = np.nonzero(gt == (j + 1))
            count[j] = np.array(ind).shape[1]
            #calculate H(Ci|T)
            if count[j] != 0:
                Hi[i] += -(count[j] / ni[i]) * math.log((count[j] / ni[i]), 2)
        H += (ni[i] / np.sum(ni)) * Hi[i]

    return H


def validateCluster(d):
    K = [5]
    sumF = 0
    sumH = 0
    t = 0
    for k in K:
        print "\n\nK = " + str(k)
        dir = d + str(k) + '/'
        for root, dirs, filenames  in os.walk(dir):
            for f in filenames:
                segmentation = np.load(dir + f)
                segmentation = segmentation.astype(np.int)
                filename = f.split('.')[0]
                plt.imsave(d + 'Segmentation Images/' + filename + '.jpg', segmentation)
                print "\nFilename: " + filename
                gt_file = './groundTruth/test/' + filename + '.mat'
                mat = sio.loadmat(gt_file)  # load mat file
                gt_size = mat['groundTruth'].size
                for i in range(0, gt_size):
                    print "GroundTruth #" + str(i + 1)
                    gt = mat['groundTruth'][0, i]   # fetch groundTruth
                    gtSegmentation = gt['Segmentation'][0][0]
                    plt.imsave(d + 'Segmentation Images/' + filename + 'gt' + str(i) + '.jpg' , gtSegmentation)
                    #F = Fmeasure(segmentation, gtSegmentation)
                    #sumF += F
                    #print "Fmeasure = " + str(F)
                    #H = ConditionalEntropy(segmentation, gtSegmentation)
                    #sumH += H
                    #print "Entropy = " + str(H)
                    t += 1

    avF = sumF / t
    avH = sumH / t
    print 'average Fmeasure = ' + avF
    print 'average Conditional Entropy = ' + avH

print 'Kmeans Validation:\n'
dir = './images1/K-means Segmentation/'
validateCluster(dir)

print 'Normalized Cut Validation:\n'
dir = './images1/NormCut Segmentation/KNN/'
validateCluster(dir)
