'''
Different functions for the final project of CSE546
for predicting words from fMRI images
Authors: Stella Stylianidou, Tomasz Sakrejda
'''
import scipy.io as io
import LassoClass as lassoSolver
from pylab import *
from scipy import sparse
import numpy as np
from fmridataloader import *
from sklearn.decomposition import PCA
import test_suite




def withPCA (dimensions):
    '''
    It finds the principal components of fmri_train and keeps the
    number of components given in dimensions. It then runs lasso on every
    semantic feature for a list of lambdas from 80 to 120 and keeps the the
    w with the least RMSE on the validation data. It returns the accuracy on
    the test data, the best weights and the pca fit. It also saves w on a file.
    :param dimensions: number of dimensions for the principal components
    :return: accuracy, bestw, the the pca fit
    '''
    pca = PCA(n_components=dimensions)
    pca.fit(fmri_train)
    xtrainpcaed= pca.transform(fmri_train)
    xtrainPCA = sparse.csc_matrix (xtrainpcaed)
    xtest = pca.transform (fmri_test)
    num_features = ytrain.shape[1]
    d = xtrainPCA.shape[1]
    ntotdata = xtrainPCA.shape[0]
    ntrain = 250 # number of data to be trained on, rest are used as cross validation
    bestw = np.zeros([num_features,d])
    accuracy = np.zeros(d)
    lasso = lassoSolver.LassoClass()
    lambda_list = list(range(80,120)) # list of lambdas to use
    for i in range(num_features):
      print ('looking at feature ', i)
      bestw[i,:]  = lasso.descendingLambda(ytrain[0:ntrain,i].reshape(ntrain,1), xtrainPCA[0:ntrain,:], ytrain[ntrain:,i].reshape(ntotdata-ntrain,1), xtrainPCA[ntrain:,:], lambda_list).reshape(d)
    wfile = "allwallfeatures_pca300_lambda80_120.mtx" # name of w file to save as
    io.mmwrite(wfile, bestw)
    test_suite.main(wfile,wordid_train,wordid_test,wordfeature_std,xtest)
    return [accuracy,bestw, pca]



def findlambda_crossvalidation():
    '''
    Cross validation using 60 samples for each validation set. Take median lambda.
    :return: l_cross_val
    '''
    num_features = ytrain.shape[1]
    d = xtrain.shape[1]
    ntotdata = xtrain.shape[0]
    n_crossvalid = 60
    bestw = np.zeros([num_features,d])
    lasso = lassoSolver.LassoClass()
    l_cross_val = []
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
    '''
    Finds the best lambda for the first semantic feature,
    using the last 50 datapoints for cross validation.
    :return: l_best round : the lambda with the least rmse on the cross validation set.
    '''
    d = xtrain.shape[1]
    ntrain = 250
    ntotdata = xtrain.shape[0]
    lasso = lassoSolver.LassoClass()
    l_best = lasso.descendingLambdaFromMax(ytrain[0:ntrain,1].reshape(ntrain,1), xtrain[0:ntrain,:], ytrain[ntrain:,1].reshape(ntotdata-ntrain,1), xtrain[ntrain:,:])
    l_best_round = int(round(l_best))
    print (l_best_round)
    return l_best_round


def run():
    '''
    It finds the best lambda for the first semantic feature and then
    around that value (+/-5) finds the best lambda for the rest
    of the semantic features
    :return: bestw
    '''
    num_features = ytrain.shape[1]
    d = xtrain.shape[1]
    ntrain = 250
    ntotdata = xtrain.shape[0]
    bestw = np.zeros([num_features,d])
    lasso = lassoSolver.LassoClass()
    l_best_round = findlambda_50validation()
    lambda_list = list(range(l_best_round + 5,  l_best_round - 5))
    for i in range(num_features):
      print ('looking at feature ', i)
      bestw[i,:]  = lasso.descendingLambda(ytrain[0:ntrain,i].reshape(ntrain,1), xtrain[0:ntrain,:], ytrain[ntrain:,i].reshape(ntotdata-ntrain,1), xtrain[ntrain:,:], lambda_list).reshape(d)
    io.mmwrite("allwallfeatures_nov17.mtx", bestw)
    return bestw

def main():
    print('hello from fmri starter type run () or findlambda_crossvalidation()')


if __name__ == '__main__':
    main()
