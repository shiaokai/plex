import settings
import numpy as np
import multiprocessing as mp

import sys
import os
import pdb
import cv2
import matplotlib.pyplot as plt
import tempfile
import cPickle

from char_det import CharDetectorBatch
from wordspot import WordSpot
from word_det_old import WordDetectorBatch, ComputeWordFeatures
from evaluation import EvaluateWordDetection
from time import time

sys.path.append(settings.libsvm_path)
import svmutil as svm

def CrossValidateAlpha(num_procs=1):
    img_dir=settings.img_train_dir
    lex_dir = settings.lex0_train_dir
    gt_dir = settings.img_train_gt_dir

    # set up temporary output dir
    #temp_char_dir = tempfile.mkdtemp()
    temp_char_dir = 'cache/chars'
    #temp_word_dir = tempfile.mkdtemp()
    temp_word_dir = 'cache/words'
    
    # call batch chardetector
    # first run character classification in batch
    """
    clf_path = settings.char_clf_name
    with open(clf_path,'rb') as fid:
        rf = cPickle.load(fid)
    CharDetectorBatch(img_dir, temp_char_dir, rf, settings.canon_size,
                      settings.alphabet_master, detect_idxs=settings.detect_idxs,
                      min_height=settings.min_height,
                      min_pixel_height=settings.min_pixel_height,
                      score_thr=settings.score_thr, num_procs=num_procs,
                      case_mapping=settings.case_mapping,
                      overlap_thr=settings.overlap_thr)
    """
    alpha_range = np.linspace(0.05, .95, 10)
    #alpha_range = [.15]
    best_alpha = -1
    best_recall = -1
        
    for alpha in alpha_range:
        start_time = time()
        WordDetectorBatch(img_dir, temp_char_dir, temp_word_dir,
                          alpha, settings.max_locations, settings.overlap_thr,
                          num_procs, lex_dir, apply_word_nms=False)
        print "Word detector time = ", time() - start_time
        eval_results=EvaluateWordDetection(gt_dir, temp_word_dir)
        gt_results = eval_results[0]
        dt_results = eval_results[1]
        precision = eval_results[2]
        recall = eval_results[3]
        thrs = eval_results[4]
        print 'Alpha = %f produced recall = %f' % (alpha, recall[-1])
        if recall[-1] > best_recall:
            best_recall = recall[-1]
            best_alpha = alpha
        #shutil.rmtree(temp_word_dir)

    WordDetectorBatch(img_dir, temp_char_dir, temp_word_dir,
                      best_alpha, settings.max_locations, settings.overlap_thr,
                      num_procs, lex_dir, apply_word_nms=False)

    eval_results=EvaluateWordDetection(gt_dir, temp_word_dir, img_dir=img_dir,
                                       create_visualization=False)

    gt_results = eval_results[0]
    dt_results = eval_results[1]
    precision = eval_results[2]
    recall = eval_results[3]
    thrs = eval_results[4]

    print 'Best alpha = %f produced recall = %f' % (best_alpha, best_recall)
    return best_alpha
        
def TrainWordClassifier(alpha, num_procs=1):
    img_dir = settings.img_train_dir
    lex_dir = settings.lex5_train_dir
    gt_dir = settings.img_train_gt_dir

    # word detector params
    max_locations = settings.max_locations
    overlap_thr = settings.overlap_thr
    
    temp_char_dir = 'cache/chars'
    temp_word_dir = 'cache/words'

    """
    clf_path = settings.char_clf_name
    with open(clf_path,'rb') as fid:
        rf = cPickle.load(fid)
    CharDetectorBatch(img_dir, temp_char_dir, rf, settings.canon_size,
                      settings.alphabet_master, detect_idxs=settings.detect_idxs,
                      min_height=settings.min_height,
                      min_pixel_height=settings.min_pixel_height,
                      score_thr=settings.score_thr, num_procs=num_procs,
                      case_mapping=settings.case_mapping,
                      overlap_thr=settings.overlap_thr)
    """

    WordDetectorBatch(img_dir, temp_char_dir, temp_word_dir,
                      alpha, settings.max_locations, settings.overlap_thr,
                      num_procs, lex_dir, apply_word_nms=False)

    eval_results=EvaluateWordDetection(gt_dir, temp_word_dir, img_dir=img_dir,
                                       create_visualization=False)
    gt_results = eval_results[0]
    dt_results = eval_results[1]
    precision_before = eval_results[2]
    recall_before = eval_results[3]
    thrs = eval_results[4]

    # plot PR
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(recall_before, precision_before)
    plt.savefig("pr_before_svm.pdf")

    Y = []
    X = []
    for dt_result1 in dt_results:
        for dt_item in dt_result1:
            word_score = dt_item[3]
            char_bbs = dt_item[4]
            features = ComputeWordFeatures(char_bbs, word_score)
            X.append(dict(zip(range(len(features)),features)))
            if dt_item[1]:
                Y.append(1)
            else:
                Y.append(-1)
            # compute features on char bbs

    # cross validate for svm parameters
    svm_model = TrainSVM(Y, X)
    svm.svm_save_model(settings.word_clf_name, svm_model)
    
def TrainSVM(Y, X, cross_validation_folds=5, sweep_c=range(-2,6)):
    best_c = -1
    best_acc = -1
    for i in range(len(sweep_c)):
        current_c = np.power(2.0,sweep_c[i])
        num_positives = float(Y.count(1))
        num_negatives = float(Y.count(-1))
        prob = svm.svm_problem(Y,X)
        param = svm.svm_parameter('-v 5 -t 0 -c %f -w-1 %f -w1 %f -q' % (current_c,
                                                                         100/num_negatives,
                                                                         100/num_positives))
        current_acc = svm.svm_train(prob, param)
        print '%f, %f' % (current_c, current_acc)
        if best_acc < current_acc:
            best_acc = current_acc
            best_c = current_c

    prob = svm.svm_problem(Y,X)
    param = svm.svm_parameter('-t 0 -c %f -w-1 %f -w1 %f -q' % (best_c,
                                                                100/num_negatives,
                                                                100/num_positives))
    svm_model = svm.svm_train(prob, param)
    p_labs, p_acc, p_vals = svm.svm_predict(Y, X, svm_model, '-q')

    return svm_model
    
def EvaluateWordspot(alpha, num_procs=1, also_show_without_svm=False):
    img_dir = settings.img_test_dir
    gt_dir = settings.img_test_gt_dir
    lex_dir = settings.lex5_test_dir
    
    temp_char_dir = 'cache/chars'
    temp_word_dir = 'cache/words'

    clf_path = settings.char_clf_name
    with open(clf_path,'rb') as fid:
        rf = cPickle.load(fid)
    # batch character detection
    CharDetectorBatch(img_dir, temp_char_dir, rf, settings.canon_size,
                      settings.alphabet_master, detect_idxs=settings.detect_idxs,
                      min_height=settings.min_height,
                      min_pixel_height=settings.min_pixel_height,
                      score_thr=settings.score_thr, num_procs=num_procs,
                      case_mapping=settings.case_mapping,
                      overlap_thr=settings.overlap_thr)

    if also_show_without_svm:
        WordDetectorBatch(img_dir, temp_char_dir, temp_word_dir,
                          alpha, settings.max_locations, settings.overlap_thr,
                          num_procs, lex_dir, svm_model=None)

        eval_results=EvaluateWordDetection(gt_dir, temp_word_dir, img_dir=img_dir,
                                           create_visualization=False)

        gt_results = eval_results[0]
        dt_results = eval_results[1]
        precision_after = eval_results[2]
        recall_after = eval_results[3]
        thrs = eval_results[4]

        # plot PR
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(recall_after, precision_after)
        plt.savefig(os.path.join(settings.fig_dir,"pr_before_svm.pdf"))


    # load svm
    svm_model = svm.svm_load_model(settings.word_clf_name)

    WordDetectorBatch(img_dir, temp_char_dir, temp_word_dir,
                      alpha, settings.max_locations, settings.overlap_thr,
                      num_procs, lex_dir, svm_model=svm_model)

    eval_results=EvaluateWordDetection(gt_dir, temp_word_dir, img_dir=img_dir,
                                       create_visualization=False)

    gt_results = eval_results[0]
    dt_results = eval_results[1]
    precision_after = eval_results[2]
    recall_after = eval_results[3]
    thrs = eval_results[4]

    # plot PR
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(recall_after, precision_after)
    plt.savefig(os.path.join(settings.fig_dir,"pr_after_svm.pdf"))

if __name__=="__main__":
    #alpha=CrossValidateAlpha(num_procs=6)
    alpha=.15
    #TrainWordClassifier(alpha,num_procs=6)
    EvaluateWordspot(alpha, num_procs=6, also_show_without_svm=True)
