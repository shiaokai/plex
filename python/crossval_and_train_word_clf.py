import settings
import numpy as np
import multiprocessing as mp

import sys
import os
import pdb
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import cPickle

from char_det import CharDetectorBatch
from word_det import WordDetectorBatch
from evaluation import EvaluateWordDetection
from time import time
from svm_helpers import TrainSvmLinear2, TrainSvmPoly2, ComputeWordFeatures

sys.path.append(settings.libsvm_path)
import svmutil as svm

def PrecomputeCharacterDetections(temp_char_dir='cache/chars'):
    img_dir = settings.img_train_dir
    num_procs=settings.n_procs

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


def CrossValidateAlpha(temp_char_dir='cache/chars'):
    img_dir = settings.img_train_dir
    lex_dir = settings.lex0_train_dir
    gt_dir = settings.img_train_gt_dir

    num_procs=settings.n_procs

    # set up temporary output dir
    temp_word_dir = 'cache/words'
    
    alpha_range = np.linspace(0.05, .95, 10)
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
        
def TrainWordClassifier(alpha, temp_char_dir='cache/chars'):
    img_dir = settings.img_train_dir
    lex_dir = settings.lex0_train_dir
    gt_dir = settings.img_train_gt_dir

    num_procs=settings.n_procs
    
    temp_word_dir = 'cache/words'

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

    Y = []
    X_list = []
    n_features = -1
    for dt_result1 in dt_results:
        for dt_item in dt_result1:
            word_score = dt_item[3]
            char_bbs = dt_item[4]
            features = ComputeWordFeatures(char_bbs, word_score)
            if n_features < 0:
                n_features = len(features)
            X_list.append(features)
            if dt_item[1]:
                Y.append(1)
            else:
                Y.append(-1)

    assert n_features > 0
    
    # scale features
    X_mat = np.vstack(X_list)
    min_vals = np.min(X_mat,axis=0)
    X_mat = X_mat - min_vals
    max_vals = np.max(X_mat,axis=0)    
    X_mat = X_mat / max_vals
    alpha_min_max = (alpha, min_vals, max_vals)
    with open(settings.word_clf_meta_name,'wb') as fid:
        cPickle.dump(alpha_min_max,fid)

    X = [dict(zip(range(n_features), x_i)) for x_i in X_mat.tolist()]

    svm_model = TrainSvmLinear2(Y, X)
    svm.svm_save_model(settings.word_clf_name, svm_model)

    svm_model_poly = TrainSvmPoly2(Y, X)
    svm.svm_save_model(settings.word_clf_poly_name, svm_model_poly)

def EvaluateWordspot(also_show_without_svm=False):

    img_dir = settings.img_test_dir
    gt_dir = settings.img_test_gt_dir
    lex_dir = settings.lex5_test_dir
    """
    img_dir = settings.img_train_dir
    gt_dir = settings.img_train_gt_dir
    lex_dir = settings.lex0_train_dir
    """

    num_procs=settings.n_procs

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

    with open(settings.word_clf_meta_name,'rb') as fid:
        alpha_min_max = cPickle.load(fid)

    alpha = alpha_min_max[0]
    min_max = (alpha_min_max[1], alpha_min_max[2])

    WordDetectorBatch(img_dir, temp_char_dir, temp_word_dir,
                      alpha, settings.max_locations, settings.overlap_thr,
                      num_procs, lex_dir, svm_model=None)

    fig = plt.figure()
    eval_results=EvaluateWordDetection(gt_dir, temp_word_dir, img_dir=img_dir,
                                       create_visualization=False,
                                       apply_word_nms=True)

    gt_results = eval_results[0]
    dt_results = eval_results[1]
    precision = eval_results[2]
    recall = eval_results[3]
    thrs = eval_results[4]

    # plot PR
    plt.hold(True)
    ax = fig.add_subplot(111)
    p1, = ax.plot(recall, precision, label='Before SVM')

    # =========================================
    # load linear svm
    # =========================================
    svm_model = svm.svm_load_model(settings.word_clf_name)

    eval_results=EvaluateWordDetection(gt_dir, temp_word_dir, img_dir=img_dir,
                                       create_visualization=False,
                                       svm_model=(svm_model, min_max),
                                       apply_word_nms=True)

    gt_results = eval_results[0]
    dt_results = eval_results[1]
    precision = eval_results[2]
    recall = eval_results[3]
    thrs = eval_results[4]

    # plot PR
    p2, = ax.plot(recall, precision, label='After SVM')

    # =========================================
    # load poly svm
    # =========================================
    svm_model_poly = svm.svm_load_model(settings.word_clf_poly_name)

    eval_results=EvaluateWordDetection(gt_dir, temp_word_dir, img_dir=img_dir,
                                       create_visualization=False,
                                       svm_model=(svm_model_poly, min_max),
                                       apply_word_nms=True)

    gt_results = eval_results[0]
    dt_results = eval_results[1]
    precision = eval_results[2]
    recall = eval_results[3]
    thrs = eval_results[4]

    # plot PR
    p3, = ax.plot(recall, precision, label='After SVM poly')

    plt.legend([p1,p2,p3], ["nosvm","linear","quad"])
    plt.savefig(os.path.join(settings.fig_dir,"pr_curves.pdf"))

if __name__=="__main__":
    PrecomputeCharacterDetections()
    alpha=CrossValidateAlpha();
    TrainWordClassifier(alpha)
    EvaluateWordspot(also_show_without_svm=True)
