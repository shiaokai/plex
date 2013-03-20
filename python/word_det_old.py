import settings

import pdb
import sys
import os
import random
import numpy as np
import cv,cv2
import cPickle
import cProfile

import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
from solve_word_old import SolveWord
from nms_old import WordBbsNms
from svm_helpers import UpdateWordsWithSvm

sys.path.append(settings.libsvm_path)
import svmutil as svm

def WordDetectorBatchWorker(job):
    char_file, save_word_path, lexicon, alpha, max_locations, overlap_thr, apply_word_nms = job 
                
    with open(char_file,'rb') as fid:
        char_bbs = cPickle.load(fid)

    word_results = WordDetector(char_bbs, lexicon, settings.alphabet_master,
                                max_locations=max_locations, alpha=alpha,
                                overlap_thr=overlap_thr, apply_word_nms=apply_word_nms)

    with open(save_word_path,'wb') as fid:
        cPickle.dump(word_results, fid)
    
def WordDetectorBatch(img_dir, char_dir, output_dir, alpha, max_locations, overlap_thr, num_procs, lex_dir, apply_word_nms=False, svm_model=None):
    # since we cannot pickle the svm model, let's apply it after word detector batch is done

    # loop through training images
    jobs = []        
    for name in os.listdir(img_dir):
        p1, ext=os.path.splitext(name)
        if ext!='.jpg':
            continue

        # check if precomp file exists
        char_file = os.path.join(char_dir, name + '.char')
        if not os.path.exists(char_file):
            print 'no results from: ', name
            continue

        lex_file = os.path.join(lex_dir, name + '.txt')
        with open(lex_file, 'r') as f:
            lexicon0 = f.readlines()
        lexicon = [l.strip() for l in lexicon0]

        # result path
        save_word_path = os.path.join(output_dir, name + '.word')
        job = (char_file, save_word_path, lexicon, alpha, max_locations, overlap_thr, apply_word_nms)
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

def WordDetector(bbs, lexicon, alphabet, max_locations=3, alpha=.5, overlap_thr=0.5,
                 svm_model=None, apply_word_nms=False):
    results = []
    for i in range(len(lexicon)):
        # force word to upper case
        word = lexicon[i].upper()
        # assume word is at least 3 characters long
        assert len(word) > 2
        word_results = SolveWord(bbs, word, alphabet, max_locations, alpha, overlap_thr)

        if not(word_results):
            continue
        
        for (word_bb, char_bbs) in word_results:
            #word_result = np.append(word_bb, [word_score, 0])
            results.append((np.expand_dims(word_bb, axis = 0), char_bbs, word))

    # check for word SVM; apply 
    if svm_model is not None:
        UpdateWordsWithSvm(svm_model, results)

    # word nms
    if apply_word_nms:
        results = WordBbsNms(results)

    return results

def ComputeWordFeatures(char_bbs, word_score):
    # 1. features based on raw scores
    #    - median/mean of character scores
    #    - min/max of character scores
    feature_vector = np.array(word_score)
    feature_vector = np.append(feature_vector, np.median(char_bbs[:,4]))
    feature_vector = np.append(feature_vector, np.mean(char_bbs[:,4]))
    feature_vector = np.append(feature_vector, np.min(char_bbs[:,4]))
    feature_vector = np.append(feature_vector, np.max(char_bbs[:,4]))
    feature_vector = np.append(feature_vector, np.std(char_bbs[:,4]))

    # 2. features based on length of string
    feature_vector = np.append(feature_vector, char_bbs.shape[0])

    # 3. features based on global layout
    #    - min/max of horizontal/vertical spaces
    x_diffs = char_bbs[1::,1] - char_bbs[0:-1,1]
    mean_width = np.mean(char_bbs[:,3])
    y_diffs = char_bbs[:,0] - np.mean(char_bbs[:,0])
    mean_height = np.mean(char_bbs[:,2])
    # standard deviation of horizontal gaps
    feature_vector = np.append(feature_vector, np.std(x_diffs / mean_width))
    # standard deviation of vertical variation
    feature_vector = np.append(feature_vector, np.std(y_diffs / mean_height))
    # max horizontal gap
    feature_vector = np.append(feature_vector, np.max(x_diffs / mean_width))
    # max vertical gap
    feature_vector = np.append(feature_vector, np.max(y_diffs / mean_height))

    # 4. features based on scale variation
    mean_scale = np.mean(char_bbs[:,2])
    feature_vector = np.append(feature_vector, np.std(char_bbs[:,2] / mean_scale))
    feature_vector = np.append(feature_vector, np.max(char_bbs[:,2] / mean_scale))

    # TODO: pairwise features?
    return feature_vector


