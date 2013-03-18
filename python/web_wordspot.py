import pdb
import os, sys
import settings
import cPickle
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import hashlib

from char_det import CharDetector
from display import OutputCharBbs, DrawCharBbs
from helpers import GetCachePath
from wordspot import WordSpot

sys.path.append(settings.libsvm_path)
import svmutil as svm

def WebWordspot(img_name, lexicon, max_locations, result_path, rf):
    img = cv2.imread(img_name)    
    svm_model = None
    #if os.path.exists(settings.word_clf_name):
    #    svm_model = svm.svm_load_model(settings.word_clf_name)

    (match_bbs, char_bbs) = WordSpot(img, lexicon, use_cache=False, img_name=img_name,
                                     max_locations=max_locations, svm_model=svm_model,
                                     rf_preload=rf)

    word_bbs = np.zeros((len(match_bbs), 6))
    for i in range(len(match_bbs)):
        word_bbs[i,:] = np.append(match_bbs[i][0],[0])

    # draw match
    DrawCharBbs(img, word_bbs, settings.alphabet_master)
    plt.savefig(result_path)


