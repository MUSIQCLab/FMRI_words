__author__ = 'Stella'

import scipy.io as io
import LassoClass as lassoSolver
from pylab import *
from scipy import sparse
import numpy as np

def run():
    dictionary = open("meta/dictionary.txt").read().splitlines()
    semantic_feature = open("meta/semantic_feature.txt").read().splitlines()
    fmri_train = io.mmread("subject1_fmri_std.train.mtx")
    fmri_train_sparse = sparse.csc_matrix (fmri_train)# sparse format
    wordfeature_std = io.mmread("word_feature_std.mtx")
    wordid_train = io.mmread("subject1_wordid.train.mtx")
    xtrain = fmri_train_sparse

    # prepare ytrain
    # careful words start from 1 - 60 not zero based!

    size_train_data = wordid_train.shape[0]
    ytrain = np.zeros([size_train_data, wordfeature_std.shape[1]])
    for i in  range (0,size_train_data):
        index = wordid_train[i][0]
        ytrain [i] = wordfeature_std [index-1]

    # get each of the semantic features separatelly and run lasso on it
    num_features = ytrain.shape[1]
    d = xtrain.shape[1]
    n_data = xtrain.shape[0]

    # only for the first question
    bestw = np.zeros([num_features,d])
    lasso = lassoSolver.LassoClass()
    lambda_list = [0.4,0.35,0.3,0.25,0.2,0.1,0.05]
    for i in range(num_features):
      bestw[i,:] = lasso.descendingLambda (ytrain[:,0].reshape(n_data,1), xtrain, lambda_list)

    return bestw
    #fmri_test = io.mmread("subject1_fmri_std.test.mtx") # sparse format
    #fmri_test_sparse = sparse.csc_matrix (fmri_test)# sparse format
    #wordid_test = io.mmread("subject1_wordid.test.mtx")
    #wordfeature_centered = io.mmread("word_feature_centered.mtx")


def main():
    print('hi')


if __name__ == '__main__':
    main()
