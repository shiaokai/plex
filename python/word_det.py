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
from solve_word import SolveWord
from nms import WordBbsNms
from svm_helpers import UpdateWordsWithSvm, ComputeWordFeatures

sys.path.append(settings.libsvm_path)
import svmutil as svm
from time import time
import shutil

def WordDetectorBatchWorker(job):
    char_file, save_word_path, lexicon, alpha, max_locations, overlap_thr, apply_word_nms = job 
                
    with open(char_file,'rb') as fid:
        char_bbs = cPickle.load(fid)

    start_time = time()
    word_results = WordDetector(char_bbs, lexicon, settings.alphabet_master,
                                max_locations=max_locations, alpha=alpha,
                                overlap_thr=overlap_thr, apply_word_nms=apply_word_nms)
    with open(save_word_path,'wb') as fid:
        cPickle.dump(word_results, fid)
    
def WordDetectorBatch(img_dir, char_dir, output_dir, alpha, max_locations, overlap_thr, num_procs, lex_dir, apply_word_nms=False, svm_model=None):
    # since we cannot pickle the svm model, let's apply it after word detector batch is done

    # clear output_dir
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

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

    start_time = time()
    if num_procs == 1:
        for job in jobs:
            WordDetectorBatchWorker(job)
    else:
        print 'using ', num_procs, ' processes to work on ', len(jobs), ' jobs.'
        pool=mp.Pool(processes=num_procs)
        pool.map_async(WordDetectorBatchWorker, jobs)
        pool.close()
        pool.join()
    print "Total word detector time: ", time() - start_time

def WordDetector(bbs, lexicon, alphabet, max_locations=3, alpha=.5, overlap_thr=0.5,
                 svm_model=None, apply_word_nms=False, timeout=120):
    results = []
    start_time = time()    
    for i in range(len(lexicon)):
        # force word to upper case
        word = lexicon[i].upper()
        # assume word is at least 3 characters long
        assert len(word) > 2
        word_results = SolveWord(bbs, word, alphabet, max_locations, alpha, overlap_thr)
        if (time() - start_time) > timeout:
            print "TIMED OUT WORD DET!"
            break

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

