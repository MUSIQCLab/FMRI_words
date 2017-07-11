'''
Different functions for the final project of CSE546
for predicting words from fMRI images
Authors: Stella Stylianidou, Tomasz sakrejda
'''
import scipy.io as io
import scipy.stats as stats
import numpy as np
from fmridataloader import *
from sklearn.decomposition import PCA
import test_suite
import matplotlib.pyplot as plt


# least squares calculation using the pseudo - inverse
def least_squares (X,Y):
    pseudo_inv_X = np.linalg.pinv (X)
    w_least_squares = np.dot(pseudo_inv_X,Y)
    return w_least_squares


def pcaData (dimensions, training_data, testing_data):
    '''
    Finds the principal components of the training for given
    amount of dimensions and returns the transformed training
    and testing data.
    :param dimensions: amount of principal components to keep
    :param training_data: data for model to be trained on
    :param testing_data: data for model to be tested on
    :return: xtrainPCA : transformed training data ,xtestPCA : tranformed testing data
    '''
    pca = PCA(n_components=dimensions)
    pca.fit(training_data)
    xtrainPCA= pca.transform(training_data)
    xtestPCA = pca.transform (testing_data)
    print(pca.explained_variance_ratio_)
    variance  = pca.explained_variance_ratio_ # plots the variance of the dimensions used
    plt.plot(variance)
    plt.show()
    return [xtrainPCA,xtestPCA]





def nonlinearFeatures (dimensions):
    '''
    Applies PCA for the given amount of dimensions. It then multiplies
    each principal component with each other principal component to create
    non linear features and finds the least squares solution.
    :param dimensions: amount of principal components to keep
    :return: [accuracy,rmsetrain,rmsetest,rmsetestwrong]
    '''
    [xtrainPCA,xtestPCA] = pcaData(300, fmri_train, fmri_train)
    n = xtrainPCA.shape[0]
    dimPCA = xtrainPCA.shape[1]
    xNonlinear = np.zeros((n,dimPCA + dimPCA*dimPCA/2))
    xtestNonlinear = np.zeros((xtestPCA.shape[0],dimPCA + dimPCA*dimPCA/2))
    counter = dimPCA
    xNonlinear [:,0:dimPCA] = xtrainPCA
    xtestNonlinear [:,0:dimPCA] = xtestPCA
    for i in range(dimPCA):
        for j in range(i+1,dimPCA):
            xNonlinear[:,counter] = xtrainPCA[:,i] * xtrainPCA[:,j]
            xtestNonlinear[:,counter] = xtestPCA[:,i] * xtestPCA[:,j]
            counter +=1
    num_features = ytrain.shape[1]
    d = xNonlinear.shape[1]
    ntotdata = xNonlinear.shape[0]
    bestw = np.zeros([num_features,d])
    bestw0 = np.zeros(num_features)
    accuracy = np.zeros(d)
    for i in range(num_features):
        y = ytrain[:,i].reshape(ntotdata,1)
        x = xNonlinear[:,:]
        w = least_squares (x,y)
        bestw[i,:]  = w.reshape(d)
    wfile = "w_lse_nonlinear.mtx"
    io.mmwrite(wfile, bestw)
    [accuracy] = test_suite.main(bestw,bestw0,wordid_test,wordfeature_std,xtestNonlinear)
    print(accuracy)
    [ytest,ywrong] = test_suite.prepareData (wordid_test,wordfeature_std)
    rmsetest = rmse_per_semantic_feature (xtestNonlinear,ytest,bestw)
    rmsetrain = rmse_per_semantic_feature (xNonlinear,ytrain,bestw)
    rmsetestwrong = rmse_per_semantic_feature (xtestNonlinear,ywrong,bestw)
    return [accuracy,rmsetrain,rmsetest,rmsetestwrong]



def findw_PCA_LSE (dimensions, train_data,test_data, wordid_test, ytrain, wfile = "w_lse_dim299.mtx"):
    '''
    Find number of dimensions given principal components of the data
    and using that it solves least squares to obtain the weights for each semantic
    feature. Using the weights obtained it calculates the predicted semantic features
    on the test data set and returns the accuracy on a guess out of 2 words. It also
    returnsrmse on test and training data, and on the wrong column of training data.
    :param dimensions: number of pca components
    :param train_data: training data
    :param test_data: test data
    :param wordid_test: ids of the words in the test data set
    :param ytrain: semantic features values for training data
    :param wfile: name of file where w's will be written
    :return:
    '''
    [xtrainPCA,xtestPCA] = pcaData(dimensions, train_data, test_data)
    num_features = ytrain.shape[1]
    d = xtrainPCA.shape[1]
    ntotdata = xtrainPCA.shape[0]
    bestw = np.zeros([num_features,d])
    bestw0 = np.zeros(num_features)
    accuracy = np.zeros(d)
    for i in range(num_features):
        print ('looking at feature ', i)
        y = ytrain[:,i].reshape(ntotdata,1)
        x = xtrainPCA[:,:]
        w = least_squares (x,y)
        bestw[i,:]  = w.reshape(dimensions)  
    io.mmwrite(wfile, bestw)
    [accuracy] = test_suite.main(bestw,bestw0,wordid_test,wordfeature_std,xtestPCA)
    print(accuracy)
    [ytest,ywrong] = test_suite.prepareData (wordid_test,wordfeature_std)
    rmsetest = rmse_per_semantic_feature (xtestPCA,ytest,bestw)
    rmsetrain = rmse_per_semantic_feature (xtrainPCA,ytrain,bestw)
    rmsetestwrong = rmse_per_semantic_feature (xtestPCA,ywrong,bestw)
    return [accuracy,rmsetrain,rmsetest,rmsetestwrong]


def word_ranking (x,w,wordid_true):
    '''
    for each x datapoint (fmri image) it calculated the predicted
    semantic features value. It then finds the rmse of the predicted semantic
    features value and each other word in the data set (y_word - x.w) ^ 2.
    Each word is ranked according to this value.
    :param x: voxels values (or PCAed x datapoint)
    :param w: weight of our model
    :param wordid_true: id of the correct word
    :return: ranks
    '''
    ntest = x.shape[0]
    num_words = wordfeature_std.shape[0]
    ranks = []
    for i in range(ntest):
        x_current = x[i]
        print('word at location : ', i)
        rmse_per_word = []
        for j in range (num_words): # go through every single word
            y = wordfeature_std [j] #semantic features for jth word.
            rmse = rmse_per_semantic_feature (x_current,y,w)
            rmse_per_word.append(rmse)
        sorted_indexes = stats.rankdata(rmse_per_word) #note argsort returns indices that would sort array
        correct_word = wordid_true[i] - 1
        rank_of_word = sorted_indexes [correct_word]
        ranks.append(rank_of_word)
    print(ranks)
    return ranks
    

def two_class_rankings(x,w,wordid_test):
    '''
    :param x:
    :param w:
    :param wordid_test:
    :return:
    '''
    wd_true = wordid_test[:,0]
    wordid_false = wordid_test[:,1]
    true_ranks = word_ranking(x,w,wd_true)
    false_ranks = word_ranking(x,w,wordid_false)
    incorrect_tests =  [i for i in range(60) if my_true_ranks[i] > my_false_ranks[i]]
    incorrect_true_ids = [int(wordid_test[i][0] - 1) for i in incorrect_tests] #compensate for wordids numbered from 1 to 60
    incorrect_false_ids = [int(wordid_test[i][1] - 1) for i in incorrect_tests]
    incorrect_true_words = [dictionary[i] for i in incorrect_true_ids]
    incorrect_false_words = [dictionary[i] for i in incorrect_false_ids]
    true_false_words = [(incorrect_true_words[i],incorrect_false_words[i]) for i in range(len(incorrect_true_words))]
    return [true_ranks, false_ranks, true_false_words]


def differentPCADimensions ():
    '''
    It goes through different pca dimensions and calculates the w
    accuracy and rmse for test and training data. It plots rmse and accuracy
    versus pca dimensions.
    '''
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
        [accuracy,rmsetrain,rmsetest,rmsetestwrong] = findw_PCA_LSE (i, fmri_train, fmri_test, wordid_test, ytrain )
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


def testing (w,xtrain,ytrain,xtest,ytest):
    '''
    Runs the different testing functions we have
    :param w: weights
    :param xtrain: training data
    :param ytrain: target training data
    :param xtest: test data
    :param ytest: target test data
    '''
    rmseTrain = rmse_per_semantic_feature (xtrain,ytrain,w)
    rmseTest = rmse_per_semantic_feature (xtest,ytest,w)
    train_line, = plt.plot(rmseTrain,  label='Training Data')
    test_line, = plt.plot(rmseTest, label='Testing Data')
    plt.xlabel ("semantic feature") # rmse per semantic feature
    plt.ylabel("RMSE")
    plt.legend(handles=[train_line, test_line])
    plt.show()
    differentPCADimensions ()


def findBadWords ():
    '''
    Plots different information about the words that were incorrectly guessed
    such as the semantic features rmse's and counts of words versus the semantic
    feature at which they didn't perform well
    :return: [rmsecorrect,rmseincorrect]
    '''
    w = io.mmread ("w_lse_dim299.mtx")
    [xtrainPCA,xtestPCA] = pcaData (299, fmri_train, fmri_test)
    [yright,ywrong] = test_suite.prepareData (wordid_test,wordfeature_std)
    [guessed_words, percentage] = test_suite.word_guesser(xtestPCA, w, 0, yright, ywrong)
    indexes_incorrect_guesses = np.where(guessed_words == False)[0].tolist()
    n = yright.shape[0]
    f = yright.shape[1]
    rmsecorrect = np.zeros((n,f))
    rmseincorrect = np.zeros((n,f))
    j = 0
    for j in range(len(indexes_incorrect_guesses)):
        i = indexes_incorrect_guesses[j]
        ypredict = xtestPCA[i].dot(w.T)
        correct_word = dictionary[int(wordid_test[i][0] - 1)]
        ycorrect = yright[i]
        rmsecorrect[j] = np.sqrt(np.square(ycorrect - ypredict))
        incorrect_word = dictionary[int(wordid_test[i][1] - 1)]
        yincorrect = ywrong[i]
        rmseincorrect[j] = np.sqrt(np.square(yincorrect - ypredict))
        print(correct_word, sum(rmsecorrect[j])/f,incorrect_word,sum(rmseincorrect[j])/f)
    correct_line, = plt.plot(rmsecorrect[0],  label='bear')
    incorrect_line, = plt.plot(rmseincorrect[0],  label='airplane')
    plt.legend(handles=[correct_line,incorrect_line])
    plt.xlabel('semantic feature')
    plt.ylabel('RMSE')
    plt.show()
    difference_rmse = rmsecorrect - rmseincorrect # if rmsecorrect > rmseincorrect then this is bad
    indexes_bad_rmse = np.where(difference_rmse>0)
    plt.hist(indexes_bad_rmse [1],range(f))
    plt.xlabel('semantic feature')
    plt.ylabel('Counts of Words')
    plt.show()
    differences_summed = np.sum (difference_rmse,axis = 0)
    plt.plot(differences_summed)
    plt.show()
    return [rmsecorrect,rmseincorrect]


def rmse_per_semantic_feature (x,y,w):
    '''
    Root mean square error per semantic feature
    :param x: data
    :param y: target values
    :param w: weights
    :return: root mean square error
    '''
    y_predict = x.dot(w.T)
    n = y_predict.shape[0]
    rmse = np.sum(1./n * np.square(y_predict - y),axis = 0) # rmse per semantic feature
    return rmse


def drop_10_words(fmri_train,wordid_train, ytrain):
    '''
    Drops 10 words and trains the model on the rest of the words
    :param fmri_train: training data set
    :param wordid_train: word id training data set
    :param ytrain: training target values
    :return: short_fmri_train, short_wordid_train, short_ytrain
    '''
    words_to_keep = 20
    num_train_words = 5 * words_to_keep
    short_wordid_train = []
    short_fmri_train = np.ones((num_train_words,21764))
    short_ytrain = ytrain
    short_ytrain = np.ones((num_train_words,218))
    drop_short = [i[0] if i[0] <= words_to_keep else -1 for i in wordid_train]
    q = 0
    for i in range(num_train_words):
        while drop_short[q] == -1:
            q += 1
        short_fmri_train[i] =  fmri_train[q]
        short_ytrain[i] = ytrain[q]
        short_wordid_train.append(wordid_train[q])
        q += 1
    [xtrainPCA,xtestPCA] = pcaData (num_train_words, fmri_train, fmri_test)
    wfile = "50_word_w_lse_dim299.mtx"
    [accuracy,rmsetrain,rmsetest,rmsetestwrong] = findw_PCA_LSE(num_train_words, short_fmri_train,fmri_test,wordid_test, short_ytrain, wfile = wfile)
    return short_fmri_train, short_wordid_train, short_ytrain


def main():
    print('hello from fmri starter type run () or findlambda_crossvalidation()')
    nonlinearFeatures (10)


if __name__ == '__main__':
    main()


