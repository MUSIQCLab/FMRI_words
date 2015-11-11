__author__ = 'Stella'

import numpy as np


def LassoClass():

    def __init__(self):
	    self.delta = 0.001

    def set_delta(self,value):
        self.delta = value


    def efficient_cord_descent_w (y, X, l_reg, w_old = None, w_0 = None):
        condition = True
        d = X.shape[1]
        a = 2 * np.sum(np.square(X.toarray()),axis=0)
        print (delta)

        if (w_old == None):
            w_old = np.zeros(d)

        if (w_0 == None):
            w_0 = 0.0

        while condition:
            w_new = np.copy (w_old)
            y_hat = X.dot(w_old) + w_0
            w_0 = estimate_w0 (y_hat,y, w_0)
            y_hat = estimate_y (y, y_hat)
            for k in range(0,d):
                w_new[k] = estimate_wk (k, y_hat, y, w_new, X, l_reg, a)
                y_hat = estimate_y_from_w (k, y_hat, w_old, w_new, X)
            condition = no_convergence(w_new,w_old)
            w_old = np.copy(w_new)
        return (w_0,w_new,y_hat)



'''
Checks for covergence between w_new and w_old
'''
def no_convergence(w_new,w_old):
    for i in range(0,w_new.shape[0]):
        if math.fabs(w_new[i] - w_old[i]) > delta:
            return True
    return False

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
