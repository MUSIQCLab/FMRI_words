__author__ = 'Stella'

import scipy.io as io
import numpy as np
from fmridataloader import *
from sklearn.decomposition import PCA
import test_suite


def least_squares (X,Y):
    pseudo_inv_X = np.linalg.pinv (X)
    w_least_squares = np.dot(pseudo_inv_X,Y)
    return w_least_squares


def with_PCA_LSE (dimensions):
    pca = PCA(n_components=dimensions)
    pca.fit(fmri_train)
    xtrainPCA= pca.transform(fmri_train)
    xtest = pca.transform (fmri_test)
    num_features = ytrain.shape[1]
    d = xtrainPCA.shape[1]
    ntotdata = xtrainPCA.shape[0]
    bestw = np.zeros([num_features,d])
    bestw0 = np.zeros(num_features)
    accuracy = np.zeros(d)
        # get best lambda for the first feature

    for i in range(num_features):
        print ('looking at feature ', i)
        y = ytrain[:,i].reshape(ntotdata,1)
        x = xtrainPCA[:,:]
        w = least_squares (x,y)
        bestw[i,:]  = w.reshape(ntotdata)


    wfile = "w_lse_dim300.mtx"
    io.mmwrite(wfile, bestw)
    # to read : bestw = io.mmread ("allwsmtx.mtx")
    test_suite.main(bestw,bestw0,wordid_test,wordfeature_std,xtest)

    return [accuracy,bestw, pca]

# square loss as a function of semantic features, no of pca dimensions
# ranking of words instead of guessing between two wrods - likelihood
#
# test and train

#def rmse_semantic_features (x,y,w):





def root_mean_square_loss (x,y,w):
    y_predict = x.dot(w.T)
    rmse = np.sum(np.square(y_predict - y),axis = 1)
    return rmse




def main():
    print('hello from fmri starter type run () or findlambda_crossvalidation()')
    with_PCA_LSE(300)




if __name__ == '__main__':
    main()
