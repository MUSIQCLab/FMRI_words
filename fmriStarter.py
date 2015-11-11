__author__ = 'Stella'

import scipy.io as io
#import LassoClass as lassoSolver
from pylab import *
from scipy import sparse


def main():
    dictionary = open("meta/dictionary.txt").read().splitlines()
    semantic_feature = open("meta/semantic_feature.txt").read().splitlines()
    fmri_train = io.mmread("subject1_fmri_std.train.mtx")
    fmri_train_sparse = sparse.csc_matrix (fmri_train)# sparse format

    fmri_test = io.mmread("subject1_fmri_std.test.mtx") # sparse format

    fmri_test_sparse = sparse.csc_matrix (fmri_test)# sparse format

    wordid_train = io.mmread("subject1_wordid.train.mtx")
    wordid_test = io.mmread("subject1_wordid.test.mtx")
    wordfeature_std = io.mmread("word_feature_std.mtx")
    wordfeature_centered = io.mmread("word_feature_centered.mtx")




if __name__ == '__main__':
    main()
