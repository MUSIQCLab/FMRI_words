# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:55:13 2015
@author: Stella,Tomasz
"""

#go through each fmri in the wordid_test and associate its semantic features

import numpy as np
import scipy.io as spio
from fmridataloader import *

def main ():
    myw = spio.mmread('allw-bestlambda193.mtx')

    num_train = len(wordid_train)
    num_test = len(wordid_test)
    n_semantic = 218
    sem_features_train = np.zeros((num_train,n_semantic))
    sem_features_test_true = np.zeros((num_test,n_semantic))
    sem_features_test_false = np.zeros((num_test,n_semantic))

    for i in range(num_train):
        sem_features_train[i] = wordfeature_std[wordid_train[i][0] - 1]

    for i in range(num_test):
        sem_features_test_true[i] = wordfeature_std[wordid_test[i,0] - 1]
        sem_features_test_false[i] = wordfeature_std[wordid_test[i,1] - 1]

    guessed_words = word_guesser(fmri_test_sparse, myw, sem_features_test_true, sem_features_test_false)
    print('the array of guesssed words is:', guessed_words)
    print ('words guessed correctly:',sum(guessed_words))
    print ('percentage guessed correctly:', sum(guessed_words)/guessed_words.size)
    return guessed_words



def tester(fmri_data, weights, sem_features): #
    predict = np.dot(fmri_data,weights.T)
    error = sem_features - predict
    sum_sqr_err = np.trace(np.dot(error,error.T)) #only the diagonal entries are what we want
    return sum_sqr_err
    # make the prediction for all the fmri's
    # calculate the squared error on each semantic feature and sum. Can possibly
    # be written as a dot product (probably).



# sem_feat_true: Semantic featutes of trues in test data
# sem_feat false: Semantic featutes of falses in test data
def word_guesser(fmri_data, weights, sem_feat_true, sem_feat_false):
    predict = fmri_data.dot(weights.T)
    num_test = len(sem_feat_true)
    test_list = np.zeros(num_test) #return values
    
    for i in range(num_test):
        rmse_false = sum(np.square(predict[i]-sem_feat_false[i]))
        rmse_true = sum(np.square(predict[i]-sem_feat_true[i]))
        if rmse_true < rmse_false:
            test_list[i] = 1
        else:
            test_list[i] = 0
    return test_list


if __name__ == '__main__':
    main()
