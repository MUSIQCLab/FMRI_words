# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:55:13 2015
@author: Stella,Tomasz
"""

#go through each fmri in the wordid_test and associate its semantic features

import numpy as np
import scipy.io as spio
#from fmridataloader import *



def prepareData (wordid_test,wordfeature_std):
    num_test = wordid_test.shape[0]
    n_semantic = wordfeature_std.shape[1]
    sem_features_test_true = np.zeros((num_test,n_semantic))
    sem_features_test_false = np.zeros((num_test,n_semantic))
    for i in range(num_test):
        sem_features_test_true[i] = wordfeature_std[wordid_test[i,0] - 1]
        sem_features_test_false[i] = wordfeature_std[wordid_test[i,1] - 1]
    return [sem_features_test_true,sem_features_test_false]



def accuracyPerFeature (weights,w0,wordid_test,wordfeature_std,fmri_test_data) :
    [sem_features_test_true,sem_features_test_false] = prepareData (wordid_test,wordfeature_std)
    predict = fmri_test_data.dot(weights.T) + w0
    num_test = len(sem_features_test_true)
    n_semantic = sem_features_test_true.shape[1]
    accuracy = np.zeros(n_semantic) #return values

    for j in range (1,n_semantic+1):
        rmse_true = np.sum(np.square(predict[:,0:j] - sem_features_test_true[:,0:j]),axis = 1)
        rmse_false = np.sum(np.square(predict[:,0:j] - sem_features_test_false[:,0:j]),axis = 1)
        test_list = rmse_true<rmse_false
        accuracy[j-1] = sum(test_list)/num_test

    plt.plot(accuracy)
    plt.ylabel("Accuracy")
    plt.xlabel("Semantic Features")
    plt.show()
    return accuracy



def main_file (wfile,w0file,wordid_test,wordfeature_std,fmri_test_data):
    w = spio.mmread(wfile)
    w0 = spio.mread(w0file)
    main  (w,w0,wordid_test,wordfeature_std,fmri_test_data)

def main (w,w0,wordid_test,wordfeature_std,fmri_test_data):
    [sem_features_test_true,sem_features_test_false] = prepareData (wordid_test,wordfeature_std)
    [guessed_words, percentage] = word_guesser(fmri_test_data, w, w0, sem_features_test_true, sem_features_test_false)
    #print('the array of guesssed words is:', guessed_words)
    #print ('words guessed correctly:',sum(guessed_words))
    print ('percentage guessed correctly:',percentage)
    return [percentage]



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
def word_guesser(fmri_data, weights, w0, sem_features_test_true, sem_features_test_false):
    predict = fmri_data.dot(weights.T)+w0
    rmse_true = np.sum(np.square(predict - sem_features_test_true),axis = 1)
    rmse_false = np.sum(np.square(predict - sem_features_test_false),axis = 1)
    test_list = rmse_true<rmse_false
    print "wordguesser:test_list"
    return [test_list, sum(test_list)/float(test_list.size)]


#if __name__ == '__main__':
 #   main()
