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
from display import DrawWordResults, DebugCharBbs
from helpers import GetCachePath
from wordspot import WordSpot

sys.path.append(settings.libsvm_path)
import svmutil as svm

def WebWordspot(img_name, lexicon, max_locations, result_path, rf, svm_model=None,
                debug_img_name=None):
    img = cv2.imread(img_name)    
    (match_bbs, char_bbs) = WordSpot(img, lexicon, use_cache=True, img_name=img_name,
                                     max_locations=max_locations, svm_model=svm_model,
                                     rf_preload=rf)

    DrawWordResults(img, match_bbs)
    plt.savefig(result_path)

    DebugCharBbs(img, char_bbs, settings.alphabet_master, lexicon)
    debug_path = result_path+'_dbg.png' 
    plt.savefig(debug_path)
    print debug_path
