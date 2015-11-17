# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:55:13 2015
@author: Stella,Tomasz
"""
import scipy.io as io
import LassoClass as lassoSolver
from pylab import *
from scipy import sparse
import numpy as np
from fmridataloader import *

# run pca to reduce dimensionality?

# questions for ta
# 1. amount of data / using some for validation
# 2. shooting? necessary? other ideas?
# 3. lambda range
# 4. is there a linear relationship? should we be transformting features?
# 5. centered data and standardized. which one to use? when is std important?



# ideas
# shuffle data before you run them
# 1. par for loop
# 2. greedy subset selection
# 3. predict the word directly - show it's worse
# shooting?
# when it is better - predict word directly

def findBestLambda(feature):
    num_features = ytrain.shape[1]
    d = xtrain.shape[1]
    ntrain = 250
    ntotdata = xtrain.shape[0]
    lasso = lassoSolver.LassoClass()
    lasso.set_error_decrease_limit(0)
    # find the best lambda for the first feature
    l_best = lasso.descendingLambdaFromMax (ytrain[0:ntrain,feature].reshape(ntrain,1), xtrain[0:ntrain,:], ytrain[ntrain:,feature].reshape(ntotdata-ntrain,1), xtrain[ntrain:,:])
    l_best_round = int(round(l_best))



def run():
    # load data
    dictionary = open("meta/dictionary.txt").read().splitlines()
    semantic_feature = open("meta/semantic_feature.txt").read().splitlines()
    fmri_train = io.mmread("subject1_fmri_std.train.mtx")
    fmri_train_sparse = sparse.csc_matrix (fmri_train)# sparse format
    wordfeature_std = io.mmread("word_feature_std.mtx")
    wordid_train = io.mmread("subject1_wordid.train.mtx")
    xtrain = fmri_train_sparse
    size_train_data = wordid_train.shape[0]
    ytrain = np.zeros([size_train_data, wordfeature_std.shape[1]])
    for i in  range (0,size_train_data):
        index = wordid_train[i][0]
        ytrain [i] = wordfeature_std [index-1]

    # get each of the semantic features separately and run lasso on it
    num_features = ytrain.shape[1]
    d = xtrain.shape[1]
    ntrain = 250
    ntotdata = xtrain.shape[0]

    bestw = np.zeros([num_features,d])
    lasso = lassoSolver.LassoClass()

    # find the best lambda for the first feature
    l_best = lasso.descendingLambdaFromMax (ytrain[0:ntrain,1].reshape(ntrain,1), xtrain[0:ntrain,:], ytrain[ntrain:,1].reshape(ntotdata-ntrain,1), xtrain[ntrain:,:])
    l_best_round = int(round(l_best))

    lambda_list = list(range(l_best_round - 5,  l_best_round+ 5))


    for i in range(num_features):
      print ('looking at feature ', i)
      bestw[i,:]  = lasso.descendingLambda(ytrain[0:ntrain,i].reshape(ntrain,1), xtrain[0:ntrain,:], ytrain[ntrain:,i].reshape(ntotdata-ntrain,1), xtrain[ntrain:,:], lambda_list).reshape(d)

    io.mmwrite("allwallfeatures.mtx", bestw)

    return bestw


def main():
    print('hello from fmri starter type run () if you want to run')


if __name__ == '__main__':
    main()
