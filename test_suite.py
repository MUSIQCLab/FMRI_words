'''
Different functions for the final project of CSE546
for predicting words from fMRI images
Authors: Stella Stylianidou, Tomasz Sakrejda
'''
import numpy as np
import scipy.io as spio


def prepareData (wordid_test,wordfeature_std):
    '''
    Takes the two columns of the test data set, the correct
    and wrong word and finds the semantic features of each word.
    It then returns them into two separate arrays.
    :param wordid_test: id's of words in the test data
    :param wordfeature_std: semantic features of words
    :return: semantic features of correct words and semantic features of incorrect words.
    '''
    num_test = wordid_test.shape[0]
    n_semantic = wordfeature_std.shape[1]
    sem_features_test_true = np.zeros((num_test,n_semantic))
    sem_features_test_false = np.zeros((num_test,n_semantic))
    for i in range(num_test):
        sem_features_test_true[i] = wordfeature_std[wordid_test[i,0] - 1]
        sem_features_test_false[i] = wordfeature_std[wordid_test[i,1] - 1]
    return [sem_features_test_true,sem_features_test_false]



def accuracyPerFeature (weights,w0,wordid_test,wordfeature_std,fmri_test_data):
    '''
    Plots how accuracy improves by adding more semantic features
    :param weights: weights
    :param w0: bias weight
    :param wordid_test: id's of words in the test data
    :param wordfeature_std: semantic features of words
    :param fmri_test_data: test data set
    :return: the accuracy
    '''
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
    '''
    Runs the main method given a file that contains the weights to be used
    :param wfile: name of file with weights
    :param w0file: name of file with bias weight
    :param wordid_test: id's of words in the test data
    :param wordfeature_std:  semantic features of words
    :param fmri_test_data: test data set
    '''
    w = spio.mmread(wfile)
    w0 = spio.mread(w0file)
    main  (w,w0,wordid_test,wordfeature_std,fmri_test_data)

def main (w,w0,wordid_test,wordfeature_std,fmri_test_data):
    '''
    Returns the percentage of words guessed correctly
    from a choice of two words
    :param w: weights
    :param w0:
    :param wordid_test:
    :param wordfeature_std:
    :param fmri_test_data:
    :return:
    '''
    [sem_features_test_true,sem_features_test_false] = prepareData (wordid_test,wordfeature_std)
    [guessed_words, percentage] = word_guesser(fmri_test_data, w, w0, sem_features_test_true, sem_features_test_false)
    print ('percentage guessed correctly:',percentage)
    return [percentage]



def tester(fmri_data, weights, sem_features):
    '''
    Predicts the semantic feature for the data and calculates the
    root mean squared error.
    :param fmri_data: data
    :param weights: weights
    :param sem_features: semantic features
    :return: sum_sqr_err
    '''
    predict = np.dot(fmri_data,weights.T)
    error = sem_features - predict
    sum_sqr_err = np.trace(np.dot(error,error.T)) #only the diagonal entries are what we want
    return sum_sqr_err






def word_guesser(fmri_data, weights, w0, sem_features_test_true, sem_features_test_false):
    '''
    Returns the accuracy on the test data, choosing the correct word out of two.
    :param fmri_data: fmri data
    :param weights: weights
    :param w0: bias weight
    :param sem_features_test_true: Semantic features of correct word in test data
    :param sem_features_test_false: Semantic features of incorrect word in test data
    :return: the test list with 1 if the word was guessed correctly and the accuracy of the test
    '''
    predict = fmri_data.dot(weights.T)+w0
    rmse_true = np.sum(np.square(predict - sem_features_test_true),axis = 1)
    rmse_false = np.sum(np.square(predict - sem_features_test_false),axis = 1)
    test_list = rmse_true<rmse_false
    print ("wordguesser:test_list")
    return [test_list, sum(test_list)/float(test_list.size)]


             
