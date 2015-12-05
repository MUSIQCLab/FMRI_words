__author__ = 'Stella'

import scipy.io as io
import numpy as np
from fmridataloader import *
from sklearn.decomposition import PCA
import test_suite
import matplotlib.pyplot as plt

# things to do
# square loss as a function of semantic features, no of pca dimensions
# ranking of words instead of guessing between two wrods - likelihood




# least squares calculation using the pseudo - inverse
def least_squares (X,Y):
    pseudo_inv_X = np.linalg.pinv (X)
    w_least_squares = np.dot(pseudo_inv_X,Y)
    return w_least_squares


# runs pca on data given dimension returns pcafitted data
# for training and test data
def pcaData (dimensions):
    pca = PCA(n_components=dimensions)
    pca.fit(fmri_train)
    xtrainPCA= pca.transform(fmri_train)
    xtestPCA = pca.transform (fmri_test)
    return [xtrainPCA,xtestPCA]


# finds the w after it pcas the data and does least square on data
# it then returns accuracy on test data on guess out of 2 words
# and returns rmse on test and training data, and also the on the wrong golcumn of training data.
def findw_PSA_LSE (dimensions):
    [xtrainPCA,xtestPCA] = pcaData(dimensions)
    num_features = ytrain.shape[1]
    d = xtrainPCA.shape[1]
    ntotdata = xtrainPCA.shape[0]
    bestw = np.zeros([num_features,d])
    bestw0 = np.zeros(num_features)
    accuracy = np.zeros(d)
    for i in range(num_features):
        #print ('looking at feature ', i)
        y = ytrain[:,i].reshape(ntotdata,1)
        x = xtrainPCA[:,:]
        w = least_squares (x,y)
        bestw[i,:]  = w.reshape(dimensions)
    #wfile = "w_lse_dim300.mtx"
    #io.mmwrite(wfile, bestw)
    # to read : bestw = io.mmread ("allwsmtx.mtx")
    [accuracy] = test_suite.main(bestw,bestw0,wordid_test,wordfeature_std,xtestPCA)
    print(accuracy)
    [ytest,ywrong] = test_suite.prepareData (wordid_test,wordfeature_std)
    rmsetest = rmse_per_semantic_feature (xtestPCA,ytest,bestw)
    rmsetrain = rmse_per_semantic_feature (xtrainPCA,ytrain,bestw)
    rmsetestwrong = rmse_per_semantic_feature (xtestPCA,ywrong,bestw)
    return [accuracy,rmsetrain,rmsetest,rmsetestwrong]


# for each x datapoint it goes through all the words we have
# and calculates the rmse of x.w with the y of each word
# it then ranks the words according to the rmse
# it finds the rank of the 'true word' and puts it in a list for all datapoints
# at the end it prints the rank
def word_ranking (x,w,wordid_true):
    ntest = x.shape[0]
    num_words = wordfeature_std.shape[0]
    ranks = []
    for i in range(ntest):
        x_current = x[i]
        print('word at location : ', i)
        rmse_per_word = []
        for j in range (num_words): # go through every single word
            y = wordfeature_std [j]
            rmse = rmse_per_semantic_feature (x_current,y,w)
            rmse_per_word.append(rmse) # it is only one datapoint so it is already summed..
        sorted_indexes = np.argsort(rmse_per_word)
        correct_word = wordid_true[i][0] - 1
        rank_of_word = sorted_indexes [correct_word]
        ranks.append(rank_of_word)
    print(ranks)

# it goes through different pca dimensions
# calculates the w and then accuracy and rmse for test and training data
# it makes different plots about all these things

def differentPCADimensions ():
    dimension_list = list(range (1,300,10))
    ndimensions = len(dimension_list)
    accuracy_list = []
    rmsetrain_list = []
    rmsetest_list = []
    n_semantic = 218
    rmsetest_matrix = np.zeros((ndimensions,n_semantic))
    rmsetestwrong_list = []

    j = 0
    for i in dimension_list:
        print(i)
        [accuracy,rmsetrain,rmsetest,rmsetestwrong] = findw_PSA_LSE (i)
        accuracy_list.append(accuracy)
        rmsetest_matrix[j] = rmsetest
        rmsetrain_list.append(np.sum(rmsetrain)/n_semantic)
        rmsetest_list.append(np.sum(rmsetest)/n_semantic)
        rmsetestwrong_list.append(np.sum(rmsetestwrong)/n_semantic)
        j = j+1

    plt.plot(dimension_list,accuracy_list)
    plt.xlabel ("PCA Dimensions")
    plt.ylabel("Accuracy")
    plt.show()

    train_line, = plt.plot(dimension_list,rmsetrain_list,  label='Training Data')
    test_line, = plt.plot(dimension_list,rmsetest_list, label='Test Data')
    wrong_test_line, = plt.plot(dimension_list,rmsetestwrong_list, label='Wrong Test Data')
    plt.xlabel ("PCA Dimensions")
    plt.ylabel("RMSE")
    plt.legend(handles=[train_line, test_line,wrong_test_line])
    plt.show()

    plt.plot(dimension_list,rmsetest_matrix)
    plt.xlabel ("PCA Dimensions")
    plt.ylabel("RMSE")
    plt.show()

    dif = rmsetest_matrix[0] - rmsetest_matrix[j-1]
    a = dif < 0  # good semantic features should have a decreasing root mean square error with more pca dimensions
    print ("semantic features that the rmse decreased", a.nonzero())
    b = dif>0.5
    print ("semantic features that the rmse increased by more than 0.5", b.nonzero())
    bad_semantic_feature = b.nonzero()
    bad_semantic_features = bad_semantic_feature[0].tolist()





# runs all the testing things
# plots the rmse per semantic feature
# also the different pca dimensions test
def testing (w,xtrain,ytrain,xtest,ytest):

    # finds rmse for train and test per semantic feature
    rmseTrain = rmse_per_semantic_feature (xtrain,ytrain,w)
    rmseTest = rmse_per_semantic_feature (xtest,ytest,w)
    train_line, = plt.plot(rmseTrain,  label='Training Data')
    test_line, = plt.plot(rmseTest, label='Testing Data')
    plt.xlabel ("semantic feature")
    plt.ylabel("RMSE")
    plt.legend(handles=[train_line, test_line])
    plt.show()

    # tests effects of different pca dimensions
    differentPCADimensions ()




# Root mean square error per semantic feature
# Calculates the predicted y given x and w and then subtracts
# from the true y, and sums up the squares of the differences, dividing
# by the number of datapoints
def rmse_per_semantic_feature (x,y,w):
    y_predict = x.dot(w.T)
    n = y_predict.shape[0]
    rmse = np.sum(1/n * np.square(y_predict - y),axis = 0) # rmse per semantic feature
    return rmse



# main function - put here what you want to run!
def main():
    print('hello from fmri starter type run () or findlambda_crossvalidation()')
    [xtrainPCA,xtestPCA] = pcaData (300)
    [ytest,ywrong] = test_suite.prepareData (wordid_test,wordfeature_std)
    w = io.mmread ("w_lse_dim300.mtx")
    testing(w,xtrainPCA,ytrain,xtestPCA,ytest)
    word_ranking (xtrainPCA,w,wordid_train)
    word_ranking (xtestPCA,w,wordid_test)




if __name__ == '__main__':
    main()


