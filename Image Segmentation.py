import numpy as np
import os
import cv2
import scipy.io as sio
import sklearn.metrics.pairwise as pw
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Cluster import Kmeans, NormCut


def displayImageGT (f):
    path = img_dir + f                                              # get full path of file
    img = cv2.imread(path, -1)                                      # read image in path
    cv2.imshow('image', img)                                        # show groundTruth as an image
    cv2.waitKey(0)                                                  # wait for a key press
    img = cv2.resize(img, (0, 0), None, .5, .5)

    # show ground truths of image
    filename = f.split('.')[0]                                      # getting filename without
    gt_file = './groundTruth/test/' + filename + '.mat'
    mat = sio.loadmat(gt_file)                                      # load mat file
    gt_size = mat['groundTruth'].size
    print gt_size
    for i in range(0, gt_size):
        gt = mat['groundTruth'][0, i]                               # fetch groundTruth
        bound = gt['Boundaries'][0][0]                              # fetch boundaries in each groundTruth
        seg = gt['Segmentation'][0][0]
        plt.imshow(seg)
        plt.show()
        print bound
        bound = 255 - bound * 255                                   # convert numbers from (0, 1) to (0, 255)
        bound_bgr = cv2.cvtColor(bound, cv2.COLOR_GRAY2BGR)         #convert image into 3-channel (BGR)
        bound_bgr = cv2.resize(bound_bgr, (0, 0), None, .5, .5)     #resize the image into half of its size
        img = np.concatenate((img, bound_bgr), axis=1)              #concatenate the image and its ground truth
    cv2.imshow('image', img)                                        # show groundTruth as an image
    cv2.waitKey(0)                                                  # wait for a key press

#=======================================================================================================================

def KNN(D, k):
    dist = distance_matrix(D, D)

    min_dist = np.argsort(dist, axis=1)[:, 1:k+1]

    A = np.zeros([D.shape[0], D.shape[0]])

    for i in range(0, D.shape[0]):
        for j in range(0, k):
            A[i, min_dist[i, j]] = 1

    return A

#=======================================================================================================================


#display all pictures in folder test
img_dir = './images1/test/'             # path of folder test
kmeans_dir = './images1/K-means Segmentation/'
nc_dir1 = './images1/NormCut Segmentation/RBF1/'
nc_dir2 = './images1/NormCut Segmentation/RBF2/'
nc_dir3 = './images1/NormCut Segmentation/KNN/'

num = 0

for root, dirs, filenames in os.walk(img_dir):                      # get all filenames in dir
    for f in filenames:                                             # display each picture and its ground truth
        if f != 'Thumbs.db':
            #show original image
            #displayImageGT(f)
            num += 1
            filename = f.split('.')[0]
            path = img_dir + f                                      # get full path of file

            img = cv2.imread(path, -1)                              # read image in path
            print '\n\nImage #' + str(num) + ' has been read'
            data1 = img.reshape(img.shape[0] * img.shape[1], 3)
            print data1.shape

            print 'Resizing Image'
            img2 = cv2.resize(img, (0, 0), None, .25, .25)
            data2 = img2.reshape(img2.shape[0] * img2.shape[1], 3)
            print data2.shape

            print 'Starting Clustering Algorithm\n'
            K = [3, 5, 7, 9, 11]
            for k in K:
                print '\nK = ' + str(k)

                kmeans_path = kmeans_dir + str(k) + '/' + filename
                nc_path1 = nc_dir1 + str(k) + '/' + filename
                nc_path2 = nc_dir2 + str(k) + '/' + filename
                nc_path3 = nc_dir3 + str(k) + '/' + filename

                #print '\nStarting K-means Algorithm'
                #[C, segmentation] = Kmeans(data1, k)

                #out = np.matrix(segmentation).reshape(img.shape[0], img.shape[1])
                #np.save(kmeans_path, out)

                #print 'K-means Clustering has finished. The clusters are:'
                #for j in range(0, k):
                #    print 'Cluster #' + str(j)
                #    print len(C[j])

                #==========================================================================


                print '\nCreating Similarity Matrix (RBF gamma = 1)'
                A = pw.rbf_kernel(data2, gamma=1)

                print 'Starting Normalized Cut Algorithm'
                [C, segmentation] = NormCut(A, k)
                out = np.matrix(segmentation).reshape(img2.shape[0], img2.shape[1])

                np.save(nc_path1 , out)

                print 'Normalized Cut (RBF gamma = 1) has finished. The clusters are:'
                for j in range(0, k):
                    print 'Cluster #' + str(j)
                    print len(C[j])

                #============================================================================

                print '\nCreating Similarity Matrix (RBF gamma = 10)'
                A = pw.rbf_kernel(data2, gamma=10)

                print 'Starting Normalized Cut Algorithm'
                [C, segmentation] = NormCut(A, k)
                out = np.matrix(segmentation).reshape(img2.shape[0], img2.shape[1])

                np.save(nc_path2, out)

                print 'Normalized Cut (RBF gamma = 10) has finished. The clusters are:'
                for j in range(0, k):
                    print 'Cluster #' + str(j)
                    print len(C[j])

                #=============================================================================

                print '\nCreating Similarity Matrix (5-NN)'
                A = KNN(data2, 5)

                print 'Starting Normalized Cut Algorithm'
                [C, segmentation] = NormCut(A, k)
                out = np.matrix(segmentation).reshape(img2.shape[0], img2.shape[1])

                np.save(nc_path3, out)

                print 'Normalized Cut (5-NN) has finished. The clusters are:'
                for j in range(0, k):
                    print 'Cluster #' + str(j)
                    print len(C[j])
