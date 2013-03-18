import sys
import os
import cv2
import settings
import cPickle
import pdb
import matplotlib.pyplot as plt

from char_det_old import CharDetector
from word_det_old import WordDetector
from display import OutputCharBbs, DrawCharBbs
from helpers import GetCachePath, CollapseLetterCase
from time import time 

def WordSpot(img, lexicon, use_cache=False, img_name='', max_locations=3,
             svm_model=None):
    cache_bbs_path = GetCachePath(img_name)
        
    if use_cache and os.path.exists(cache_bbs_path):
        print 'Found cached character bbs'
        with open(cache_bbs_path,'rb') as fid:
            char_bbs = cPickle.load(fid)
    else:
        with open(settings.char_clf_name,'rb') as fid:
            rf = cPickle.load(fid)
        print 'Loaded character classifier'

        # run character detection to get bbs
        char_bbs = CharDetector(img, settings.hog, rf, settings.canon_size,
                                settings.alphabet_master, min_height=settings.min_height,
                                detect_idxs=settings.detect_idxs,
                                debug=True, score_thr=settings.score_thr,
                                overlap_thr=settings.overlap_thr,
                                case_mapping=settings.case_mapping)
        if use_cache:
            with open(cache_bbs_path,'wb') as fid:
                cPickle.dump(char_bbs,fid)

    start_time = time()
    word_bbs = WordDetector(char_bbs, lexicon, settings.alphabet_master,
                            max_locations=max_locations, alpha=settings.alpha,
                            overlap_thr=settings.overlap_thr, svm_model=svm_model)
    print 'Word detector time: ', time() - start_time
    return (word_bbs, char_bbs)

