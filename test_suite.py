# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:55:13 2015

@author: Tomasz
"""

#go through each fmri in the wordid_test and associate its semantic features

import numpy as np
import scipy.io as spio

myw = spio.mmread('C:\\Users\\Tomasz\\Documents\\CSE546\\fmriproject\\allwsmtx.mtx')

num_train = len(wordid_train)
num_test = len(wordid_test)

sem_features_train = np.zeros((300,218))
sem_features_test_true = np.zeros((60,218))
sem_features_test_false = np.zeros((60,218))

for i in range(num_train):
    sem_features_train[i] = wordfeature_std[wordid_train[i][0] - 1]

for i in range(num_test):
    sem_features_test_true[i] = wordfeature_std[wordid_test[i,0] - 1]
    sem_features_test_false[i] = wordfeature_std[wordid_test[i,1] - 1]    
    
def tester(fmri_data, weights, sem_features): #
    predict = np.dot(fmri_data,weights.T)
    error = sem_features - predict
    sum_sqr_err = np.trace(np.dot(error,error.T)) #only the diagonal entries are what we want
    return sum_sqr_err
    #make the prediction for all the fmri's
    #calculate the squared error on each semantic feature and sum. Can possibly
    #be written as a dot product (probably).


#sem_feat_true: Sem_feats of trues in test data
#sem_feat false: See above

def word_guesser(fmri_data, weights, sem_feat_true, sem_feat_false):
    predict = np.dot(fmri_data, weights.T)
    num_test = len(sem_feat_true)
    test_list = np.zeros(num_test) #return values
    
    for i in range(num_test): #both lengths the same
        pred = predict[i,0]
        err1 = pred - sem_feat_true[i]
        sum_sqr_err1 = np.dot(err1,err1.T)
        err2 = pred - sem_feat_false[i]
        sum_sqr_err2 = np.dot(err2,err2.T)
        if sum_sqr_err2 >= sum_sqr_err1:
            test_list[i] = 0
        else:
            test_list[i] = 1
    return test_list
            
            