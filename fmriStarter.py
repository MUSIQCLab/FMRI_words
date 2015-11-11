__author__ = 'Stella'

import scipy.io as io
import LassoClass as lassoSolver
from pylab import *
from scipy import sparse
import numpy as np

def main():
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

    print
    # get each of the semantic features separatelly and run lasso on it
    n = xtrain.shape[0]
    print(n)
    for i in range(n):
        linit = lassoSolver.calculate_max_lambda (xtrain,ytrain[:,i])
        print(i, ' ', linit)

    #fmri_test = io.mmread("subject1_fmri_std.test.mtx") # sparse format
    #fmri_test_sparse = sparse.csc_matrix (fmri_test)# sparse format
    #wordid_test = io.mmread("subject1_wordid.test.mtx")
    #wordfeature_centered = io.mmread("word_feature_centered.mtx")




if __name__ == '__main__':
    main()
