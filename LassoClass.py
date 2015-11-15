__author__ = 'Stella'

import numpy as np
import math

class LassoClass:
    def __init__(self):
        self.delta = 0.01
        self.error_decrease_limit = 0.001

    def set_delta(self,value):
        self.delta = value

    def set_error_decrease_limit (self,value):
        self.error_decrease_limit = value


    def descendingLambdaFromMax (self, ytrain, xtrain, yvalid, xvalid,w_new = None, w_0 = 0.0):
        lambda_ratio = 0.8
        d = xtrain.shape[1]

        if (w_new == None):
            w_new = np.zeros((d,1))

        #l_reg = calculate_max_lambda (xtrain,ytrain)
        error_decreases = True
        previous_error = float("inf")
        l_reg_list = []
        RMSE_train = []
        RMSE_valid = []
        nonZeros_list = []
        weights_list = []
        l_reg = calculate_max_lambda (xtrain,ytrain)
        while error_decreases:
            print("coordinate descent with lambda", l_reg, ' and using previous w')
            (w_0,w_new,y_train_hat) = self.cordDescentLasso(np.copy(ytrain), xtrain, l_reg, w_new, w_0)
            rmsetrain = root_mean_squared_error(ytrain, y_train_hat)
            rmsevalid = root_mean_squared_error(yvalid,calculate_predicted_y(xvalid,w_new,w_0) )
            nonZeros = np.count_nonzero(w_new)
            nonZeros_list.append(nonZeros)
            l_reg_list.append(l_reg)
            RMSE_train.append(rmsetrain)
            RMSE_valid.append(rmsevalid)
            weights_list.append(w_new)
            print ("lambda : ", l_reg, "rmse validation ", rmsevalid, "rmse train : ", rmsetrain, " non Zeros : ", nonZeros)
            error_decreases = (previous_error - rmsevalid) > self.error_decrease_limit
            previous_error = rmsevalid
            l_reg = l_reg * lambda_ratio

        # find best lambda - smallest validation error
        ind_best_l = np.argmin(RMSE_valid)
        print(RMSE_train)
        print(RMSE_valid)
        print(l_reg_list)
        print('best lambda is ', l_reg_list[ind_best_l])
        return (l_reg_list[ind_best_l])



    def descendingLambda (self,ytrain, xtrain, yvalid, xvalid,lambda_list, w_new = None, w_0 = 0.0):
        d = xtrain.shape[1]

        if (w_new == None):
            w_new = np.zeros((d,1))

        #l_reg = calculate_max_lambda (xtrain,ytrain)
        error_decreases = True
        previous_error = float("inf")
        l_reg_list = []
        RMSE_train = []
        RMSE_valid = []
        nonZeros_list = []
        weights_list = []

        for l_reg in lambda_list:
            print("coordinate descent with lambda", l_reg, ' and using previous w')
            (w_0,w_new,y_train_hat) = self.cordDescentLasso(np.copy(ytrain), xtrain, l_reg, w_new, w_0)
            rmsetrain = root_mean_squared_error(ytrain, y_train_hat)
            rmsevalid = root_mean_squared_error(yvalid,calculate_predicted_y(xvalid,w_new,w_0) )
            nonZeros = np.count_nonzero(w_new)
            nonZeros_list.append(nonZeros)
            l_reg_list.append(l_reg)
            RMSE_train.append(rmsetrain)
            RMSE_valid.append(rmsevalid)
            weights_list.append(w_new)
            print ("lambda : ", l_reg, "rmse validation ", rmsevalid, "rmse train : ", rmsetrain, " non Zeros : ", nonZeros)


        # find best lambda - smallest validation error
        ind_best_l = np.argmin(RMSE_valid)
        print (RMSE_valid)
        print(RMSE_train)
        print(l_reg_list)
        return (weights_list[ind_best_l])


    def cordDescentLasso (self, y, x, l_reg, w_old , w_0):
        condition = True
        d = x.shape[1]
        a = 2 * np.sum(np.square(x.toarray()),axis=0)

        while condition:
            w_new = np.copy (w_old)
            y_hat = x.dot(w_old) + w_0
            w_0 = estimate_w0 (y_hat,y, w_0)
            y_hat = estimate_y (y, y_hat)
            for k in range(0,d):
                w_new[k] = estimate_wk (k, y_hat, y, w_new, x, l_reg, a)
                y_hat = estimate_y_from_w (k, y_hat, w_old, w_new, x)
            condition = self.no_convergence(w_new,w_old)
            w_old = np.copy(w_new)

        return (w_0,w_new,y_hat)

    '''
    Checks for covergence between w_new and w_old
    '''
    def no_convergence(self,w_new,w_old):
        for i in range(0,w_new.shape[0]):
            if math.fabs(w_new[i] - w_old[i]) > self.delta:
                return True
        return False


'''
Root mean squared error from y to y_prediction
'''
def root_mean_squared_error (y_valid,y_predict):
    diff = y_valid - y_predict
    return math.sqrt(np.sum(np.square(diff))/diff.shape[0])


'''
Calculates predicted y from w
'''
def calculate_predicted_y (X,w,w_0):
    return(X.dot(w) + w_0)




'''
Estimates w_0 from y_hat
'''
def estimate_w0 (y_hat,y, w_0):
    y_dif_matrix = y - y_hat
    w_0 = (y_dif_matrix.sum() / y_dif_matrix.shape[0]) + w_0
    return w_0

'''
Estimates w_k from y_hat
'''
def estimate_wk (k, y_hat, y, w, X, l_reg, a):
    ck = 0
    y_dif = y - y_hat

    test = X.getcol(k).toarray() * w[k]
    sum = y_dif + test
    ck = 2 * X.getcol(k).transpose().dot(sum)

    if ck  < - l_reg :
        w[k] = (ck + l_reg) / a[k]
    elif ck  > l_reg :
        w[k] = (ck - l_reg) / a[k]
    else:
        w[k] = 0
    return w[k]

'''
Estimates y from y and y hat
'''
def estimate_y (y, y_hat):
    y_dif_matrix = y - y_hat
    y_new = y_hat + (y_dif_matrix.sum() / y_dif_matrix.shape[0])
    return y_new


'''
Estimates y from y hat and w
'''
def estimate_y_from_w  (k, y_hat, w_old, w_new, X):
    y_new = y_hat +  (w_new[k] -  w_old[k]) * X.getcol(k).toarray()
    return y_new

'''
Method that calculated the max lambda
used to start coordinate descent with
'''
def calculate_max_lambda (X,y):
    return 2 * np.max(np.fabs(X.transpose(True).dot(y - np.average(y))))
