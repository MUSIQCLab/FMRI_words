__author__ = 'Stella'

import scipy.io as io
import LassoClass
from pylab import *


def loadFmriData ():
    fmri_train = io.mmread("subject1_fmri_std.train.mtx").tocsc() # sparse format
    fmri_test = io.mmread("subject1_fmri_std.test.mtx").tocsc() # sparse format
    wordid_train = io.mmread("subject1_wordid.train.mtx")
    wordid_test = io.mmread("subject1_wordid.test.mtx")
    wordfeature_centered = io.mmread("word_feature_centered.mtx")
    wordfeature_std = io.mmread("word_feature_std.mtx")

    dictionary = open("meta/dictionary.txt").read().splitlines()
    semantic_feature = open("meta/semantic_feature.txt").read().splitlines()


loadFmriData()