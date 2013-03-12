import os
import cv2
import settings
import cPickle
import pdb
import matplotlib.pyplot as plt

from char_det import CharDetector
from word_det_old import WordDetector
from display import OutputCharBbs, DrawCharBbs
from helpers import GetCachePath, CollapseLetterCase

def WordSpot(img, lexicon, use_cache=False, img_name='',
             alpha=.5, max_locations=3):
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
                                settings.alphabet_master, min_height=0.03,
                                detect_idxs=settings.detect_idxs,
                                debug=True, score_thr=.1,
                                case_mapping = settings.case_mapping)
        if use_cache:
            with open(cache_bbs_path,'wb') as fid:
                cPickle.dump(char_bbs,fid)

    word_bbs = WordDetector(char_bbs, lexicon, settings.alphabet_master,
                             max_locations=max_locations, alpha=alpha)
    return (word_bbs, char_bbs)

