__author__ = 'Stella'

import scipy.io as io
from pylab import *


def loadFmriData ():
    fmri_train = io.mmread("subject1_fmri_std.train.mtx") # sparse format
    fmri_test = io.mmread("subject1_fmri_std.test.mtx")# sparse format
    wordid_train = io.mmread("subject1_wordid.train.mtx") # sparse format
    wordid_test = io.mmread("subject1_wordid.test.mtx") # sparse format
    wordfeature_centered = io.mmread("word_feature_centered.mtx") # sparse format
    wordfeature_std = io.mmread("word_feature_std.mtx") # sparse format

    dictionary = open("meta/dictionary.txt").read().splitlines()
    semantic_feature = open("meta/semantic_feature.txt").read().splitlines()
    #

loadFmriData()