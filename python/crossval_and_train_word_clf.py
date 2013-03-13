import settings
import numpy as np
import multiprocessing as mp

import os
import pdb
import cv2
import matplotlib.pyplot as plt
import tempfile
import cPickle

from char_det import CharDetectorBatch
from wordspot import WordSpot
from word_det_old import WordDetector
from evaluation import EvaluateWordDetection

def WordDetectorBatchWorker(job):
    char_file, save_path, lexicon, alpha, max_locations = job 
                
    with open(char_file,'rb') as fid:
        char_bbs = cPickle.load(fid)

    word_results = WordDetector(char_bbs, lexicon, settings.alphabet_master,
                                max_locations=max_locations, alpha=alpha)

    with open(save_path,'wb') as fid:
        cPickle.dump(word_results, fid)
    
def WordDetectorBatch(img_dir, char_dir, output_dir, alpha, max_locations, num_procs):
    # loop through training images
    jobs = []        
    for name in os.listdir(img_dir):
        p1,ext=os.path.splitext(name)
        if ext!='.jpg':
            continue

        # check if precomp file exists
        char_file = os.path.join(char_dir, name + '.char')
        if not os.path.exists(char_file):
            print "no results from: ", name
            continue

        # for tuning, just use GT as lexicon
        lex_file = os.path.join(settings.lex0_train_dir,
                                name + '.txt')
        with open(lex_file, 'r') as f:
            lexicon0 = f.readlines()
        lexicon = [l.strip() for l in lexicon0]

        # result path
        save_path = os.path.join(output_dir, name + '.word')
        job = (char_file, save_path, lexicon, alpha, max_locations)
        jobs.append(job)


    if num_procs == 1:
        for job in jobs:
            WordDetectorBatchWorker(job)
    else:
        print 'using ', num_procs, ' processes to work on ', len(jobs), ' jobs.'
        pool=mp.Pool(processes=num_procs)
        pool.map_async(WordDetectorBatchWorker, jobs)
        pool.close()
        pool.join()

def CrossValidateAlpha(num_procs=1):
    '''
    for each alpha in range [0:1:.05]:
      run wordspot using alpha
      pickle results, TODO: specify cache directory base in settings

    compute fscore
    pickle and cache alpha
    '''
    # set up temporary output dir
    #temp_char_dir = tempfile.mkdtemp()
    temp_char_dir = 'cache/chars'
        
    # character detector params
    min_height = 0.05
    score_thr = 0.1
    img_dir = settings.img_train_dir
    canon_size = settings.canon_size
    alphabet = settings.alphabet_master
    detect_idxs = settings.detect_idxs

    # word detector params
    max_locations = 5

    # call batch chardetector
    # first run character classification in batch

    """
    clf_path = settings.char_clf_name
    with open(clf_path,'rb') as fid:
        rf = cPickle.load(fid)
    CharDetectorBatch(img_dir, temp_char_dir, rf, canon_size, alphabet,
                      detect_idxs=detect_idxs, min_height=min_height,
                      score_thr=score_thr, num_procs=6, case_mapping=settings.case_mapping)
    """

    alpha_range = np.linspace(0.05, .95, 10)
    #alpha_range = [.15]
    best_alpha = -1
    best_recall = -1
    for alpha in alpha_range:

        #temp_word_dir = tempfile.mkdtemp()
        temp_word_dir = 'cache/words'
        WordDetectorBatch(settings.img_train_dir, temp_char_dir, temp_word_dir,
                          alpha, max_locations, num_procs)
        eval_results=EvaluateWordDetection(settings.img_train_gt_dir,
                                           temp_word_dir)

        gt_results = eval_results[0]
        dt_results = eval_results[1]
        precision = eval_results[2]
        recall = eval_results[3]
        thrs = eval_results[4]

        if recall[-1] > best_recall:
            best_recall = recall[-1]
            best_alpha = alpha

        #shutil.rmtree(temp_word_dir)

    print "Best alpha = %f produced recall = %f" % (best_alpha, best_recall)

    # re-run with best alpha and optionally produce debug output
    temp_word_dir = 'cache/words'
    WordDetectorBatch(settings.img_train_dir, temp_char_dir, temp_word_dir,
                      best_alpha, max_locations, num_procs)
    eval_results=EvaluateWordDetection(settings.img_train_gt_dir,
                                       temp_word_dir)

    # compute goodness of alpha: every TP match in results gets 1pt,
    # find alpha that produces most points
    #
        
if __name__=="__main__":

    CrossValidateAlpha(num_procs=6)

    '''

    run wordspot on train images with best alpha

    grab FPs and TPS and train SVM
    '''
