# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:55:13 2015
@author: Stella,Tomasz
"""

#go through each fmri in the wordid_test and associate its semantic features

import scipy.io as io
import LassoClass as lassoSolver
from pylab import *
from scipy import sparse
import numpy as np
from fmridataloader import *
from sklearn.decomposition import PCA
import test_suite

# questions
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
# when it is better - predict word directly





def withPCA (dimensions):
    pca = PCA(n_components=dimensions)
    pca.fit(fmri_train)
    xtrainpcaed= pca.transform(fmri_train)
    xtrainPCA = sparse.csc_matrix (xtrainpcaed)
    xtest = pca.transform (fmri_test)
    num_features = ytrain.shape[1]
    d = xtrainPCA.shape[1]
    ntrain = 250
    ntotdata = xtrainPCA.shape[0]

    bestw = np.zeros([num_features,d])
    lasso = lassoSolver.LassoClass()
    accuracy = np.zeros(d)
    # get best lambda for the first feature
    lambda_list = list(range(80,120))

    for i in range(num_features):
      print ('looking at feature ', i)
      bestw[i,:]  = lasso.descendingLambda(ytrain[0:ntrain,i].reshape(ntrain,1), xtrainPCA[0:ntrain,:], ytrain[ntrain:,i].reshape(ntotdata-ntrain,1), xtrainPCA[ntrain:,:], lambda_list).reshape(d)

    wfile = "allwallfeatures_pca300_lambda80_120.mtx"
    io.mmwrite(wfile, bestw)

    # to read : bestw = io.mmread ("allwsmtx.mtx")

    test_suite.main(wfile,wordid_train,wordid_test,wordfeature_std,xtest)

    return [accuracy,bestw, pca]






def findlambda_crossvalidation():
    num_features = ytrain.shape[1]
    d = xtrain.shape[1]
    ntotdata = xtrain.shape[0]
    n_crossvalid = 60
    bestw = np.zeros([num_features,d])
    lasso = lassoSolver.LassoClass()
    #k-fold the data (k = 60). Cross validation using 60 samples for each validation set. Take median lambda.
    #note that we should randomly try several different words to train on...
    l_cross_val = []

    # find the best lambda for the first feature
    for i in range(ntotdata/n_crossvalid):
        validation = range(i * n_crossvalid, (i+1)*n_crossvalid) #generates the numbers corresponding to the validation set
        train = list(set(range(ntotdata)) - set(validation)) # generates nums corresponding to test set
        ytraintrain = ytrain[train,1].reshape(len(train),1)
        xtraintrain = xtrain[train,:]
        yvalid = ytrain[validation,1].reshape(len(validation),1)
        xvalid = xtrain[validation,:]
        l_best = lasso.descendingLambdaFromMax(ytraintrain, xtraintrain, yvalid, xvalid)
        l_cross_val.append(l_best)

    print (l_cross_val)
    return l_cross_val


def findlambda_50validation():
    num_features = ytrain.shape[1]
    d = xtrain.shape[1]
    ntrain = 250
    ntotdata = xtrain.shape[0]
    lasso = lassoSolver.LassoClass()
    #get best lambda for the first feature
    l_best = lasso.descendingLambdaFromMax(ytrain[0:ntrain,1].reshape(ntrain,1), xtrain[0:ntrain,:], ytrain[ntrain:,1].reshape(ntotdata-ntrain,1), xtrain[ntrain:,:])
    l_best_round = int(round(l_best))
    print (l_best_round)
    return l_best_round


def run():
    # get each of the semantic features separatelly and run lasso on it
    #l_best = findlambda_50validation()

    num_features = ytrain.shape[1]
    d = xtrain.shape[1]
    ntrain = 250
    ntotdata = xtrain.shape[0]

    bestw = np.zeros([num_features,d])
    lasso = lassoSolver.LassoClass()

    # get best lambda for the first feature
    l_best_round = 10
    lambda_list = list(range(l_best_round + 5,  l_best_round - 5))

    #pool = Pool() #defaults to number of available CPU's
    #chunksize = 20 #this may take some guessing ... take a look at the docs to decide
    #for i in enumerate(pool.imap(Fun, product(xrange(N), xrange(N))), chunksize):
    #for i in range(num_features):
     #  print ('looking at feature ', i)
      # bestw[i,:]  = lasso.descendingLambda(ytrain[0:ntrain,i].reshape(ntrain,1), xtrain[0:ntrain,:], ytrain[ntrain:,i].reshape(ntotdata-ntrain,1), xtrain[ntrain:,:], lambda_list).reshape(d)


    for i in range(num_features):
      print ('looking at feature ', i)
      bestw[i,:]  = lasso.descendingLambda(ytrain[0:ntrain,i].reshape(ntrain,1), xtrain[0:ntrain,:], ytrain[ntrain:,i].reshape(ntotdata-ntrain,1), xtrain[ntrain:,:], lambda_list).reshape(d)

    io.mmwrite("allwallfeatures_nov17.mtx", bestw)
    # to read : neww = io.mmread ("allwsmtx.mtx")
    return bestw


def crossValidationPCA (dimensions):
    pca = PCA(n_components=dimensions)
    pca.fit(fmri_train)
    xtrainpcaed= pca.transform(fmri_train)
    xtrainPCA = sparse.csc_matrix (xtrainpcaed)
    xtest = pca.transform (fmri_test)
    num_features = ytrain.shape[1]
    d = xtrainPCA.shape[1]
    ntotdata = xtrainPCA.shape[0]

    bestw = np.zeros([num_features,d])
    lasso = lassoSolver.LassoClass()

    for i in range(num_features):
        print ('looking at feature ', i)
        bestlambda = lasso.findLambdaCrossValidation(ytrain[:,i].reshape(ntotdata,1),xtrainPCA,5)
        bestw[i,:]  = lasso.cordDescentLasso (ytrain[:,i].reshape(ntotdata,1),xtrainPCA, bestlambda)

    wfile = "allwallfeatures_crossvalidationlambda_pca300.mtx"
    io.mmwrite(wfile, bestw)
    test_suite.main(wfile,wordid_train,wordid_test,wordfeature_std,xtest)
    return [accuracy,bestw, pca]

def main():
    print('hello from fmri starter type run () or findlambda_crossvalidation()')
    crossValidationPCA(300)




if __name__ == '__main__':
    main()
