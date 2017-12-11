import numpy as np
from decimal import *
import scipy as sp


def Kmeans(D, k):
    t = 0
    e = 0.01
    mean = []
    # generate random 3D means
    print 'Generating initial means...'
    j = 0
    i = 0
    while i < k:
        step = len(D) / (k - 1)
        if list(D[j - 1, :]) not in mean:
            mean.append(list(D[j - 1, :]))
            j += step
        else:
            i -= 1
            j -= 1
        i += 1

    print "Means:\n", mean

    segmentation = np.zeros([D.shape[0],])

    # repeat till convergance
    while True:
        print 'iteration #' + str(t)

        C = []
        old_mean = mean[:]
        for j in range(0, k):
            C.append([])

        # assign data to clusters
        print 'Assigning points to clusters...'
        for x in range(0, len(D)):
            distance = []
            for i in range(0, k):
                distance.append(np.sqrt(np.sum(np.square(D[x] - mean[i]))))
            min = np.argmin(distance)
            C[min].append(D[x])
            segmentation[x] = min + 1
        # update centroids
        print 'Updating Centroids...'
        for i in range(0, k):
                mean[i] = np.mean(C[i], axis=0)
        # calculate error
        print 'Calculating Error...'
        error = []
        for i in range(0, k):
            error.append(np.sqrt(np.sum(np.square(mean[i] - old_mean[i]))))
        sum_error = np.sum(error)

        # break if tolerance is reached
        if sum_error <= e or t == 100:
            break

        t += 1
        break

    return np.array(C), np.array(segmentation)
#=======================================================================================================================


def NormCut(A, k):
    print 'Creating Degree Matrix'
    deg = np.diag(np.max([np.sum(A, axis=1), np.sum(A, axis=0)], axis=0))

    print 'Calculating L Matrix'
    L = deg - A

    print 'Calculating B Matrix'
    B = np.linalg.pinv(deg).dot(L)

    print 'Getting Eigen Values and Eigen Vectors'
    [v, w] = sp.linalg.eigh(B)

    #np.save('Eigen Values', v)
    #np.save('Eigen Vectors', w)

    #v = np.load('Eigen Values.npy')
    #w = np.load('Eigen Vectors.npy')

    print 'Sorting Eigen Values'
    sorted_args = np.argsort(v)

    print 'Selecting Eigen Vectors'
    ind = sorted_args[:k]

    U = []
    for i in range(0, k):
        U.append(w[:, ind[i]])
    U = np.array(U).T

    print 'Normalizing Chosen Eigen Vectors'
    Y = np.zeros([U.shape[0], k])
    for i in range(0, U.shape[0]):
        if np.sqrt(np.sum(np.square(U[i, :]))) != 0:
            Y[i] = (1.0 / np.sqrt(np.sum(np.square(U[i, :])))) * (U[i, :].T)

    return Kmeans(Y, k)
#=======================================================================================================================