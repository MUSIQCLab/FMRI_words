# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:16:17 2015

@author: Tomasz
"""

import scipy.io as io
from scipy import sparse
import numpy as np

dictionary = open("meta/dictionary.txt").read().splitlines()
semantic_feature = open("meta/semantic_feature.txt").read().splitlines()

print('loading fmri training and test data')

fmri_train = io.mmread("subject1_fmri_std.train.mtx")
fmri_train_sparse = sparse.csc_matrix (fmri_train)# sparse format
fmri_test = io.mmread("subject1_fmri_std.test.mtx") # sparse format
fmri_test_sparse = sparse.csc_matrix (fmri_test)# sparse format

print('loading word data')

wordid_train = io.mmread("subject1_wordid.train.mtx")
wordid_test = io.mmread("subject1_wordid.test.mtx")
wordfeature_std = io.mmread("word_feature_std.mtx")
wordfeature_centered = io.mmread("word_feature_centered.mtx")

print('getting y training data into format')


xtrain = fmri_train_sparse

size_train_data = wordid_train.shape[0]
ytrain = np.zeros([size_train_data, wordfeature_std.shape[1]])
for i in  range (0,size_train_data):
    index = wordid_train[i][0]
    ytrain [i] = wordfeature_std [index-1]
