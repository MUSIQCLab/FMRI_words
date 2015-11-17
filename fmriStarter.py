__author__ = 'Stella'

import scipy.io as io
import LassoClass as lassoSolver
from pylab import *
from scipy import sparse
import numpy as np

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
# 4. Tie togethers semantic features into bundles? 
# 5. Cross validation
# 6. Our Y-values do really look strange as stella mentioned. Should we talk about that when we give our results?

# shooting?
# when it is better - predict word directly

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
    ntrain = 250
    ntotdata = xtrain.shape[0]

    # only for the first question
    bestw = np.zeros([num_features,d])
    lasso = lassoSolver.LassoClass()
    
    #k-fold the data (k = 60). Cross validation using 60 samples for each validation set. Take median lambda.    
    #note that we should randomly try several different words to train on...
    
    l_cross_val = []
    # find the best lambda for the first feature
    for i in range(5):
        validation = range(i * 60, (i+1)*60) #generates the numbers corresponding to the validation set
        train = list(set(range(300)) - set(validation)) # generates nums corresponding to test set
        ytraintrain = ytrain[train,1].reshape(len(train),1)
        xtraintrain = xtrain[train,:]
        yvalid = ytrain[validation,1].reshape(len(validation),1)
        xvalid = xtrain[validation,:]
        l_best = lasso.descendingLambdaFromMax(ytraintrain, xtraintrain, yvalid, xvalid)
        l_cross_val.append(l_best)
    
    print "woob woob woob"
    print l_cross_val

#    l_best = lasso.descendingLambdaFromMax(ytrain[0:ntrain,1].reshape(ntrain,1), xtrain[0:ntrain,:], ytrain[ntrain:,1].reshape(ntotdata-ntrain,1), xtrain[ntrain:,:])   
#    def descendingLambdaFromMax (self, ytrain, xtrain, yvalid, xvalid,w_new = None, w_0 = 0.0):


    l_best_round = int(round(l_best))


    lambda_list = list(range(l_best_round - 5,  l_best_round+ 5))


#    for i in range(num_features):
#      print ('looking at feature ', i)
#      bestw[i,:]  = lasso.descendingLambda(ytrain[0:ntrain,i].reshape(ntrain,1), xtrain[0:ntrain,:], ytrain[ntrain:,i].reshape(ntotdata-ntrain,1), xtrain[ntrain:,:], lambda_list).reshape(d)
#
#    io.mmwrite("allwallfeatures.mtx", bestw)
#    # to read : neww = io.mmread ("allwsmtx.mtx")
#
#    return bestw
    #fmri_test = io.mmread("subject1_fmri_std.test.mtx") # sparse format
    #fmri_test_sparse = sparse.csc_matrix (fmri_test)# sparse format
    #wordid_test = io.mmread("subject1_wordid.test.mtx")
    #wordfeature_centered = io.mmread("word_feature_centered.mtx")


def main():
    print('hello from fmri starter type run () if you want to run')


if __name__ == '__main__':
    main()
