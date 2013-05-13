import settings
import multiprocessing as mp
import pdb
import os
import random
import numpy as np
import cv,cv2
import cPickle
import cProfile
import copy
import tempfile

from hog_utils import draw_hog, ReshapeHog
from char_det import CharDetector, CharDetectorBatch

from time import time
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from numpy import arange
import shutil

from display import OutputCharBbs, DrawCharBbs
from helpers import ReadAllImages
from evaluation import EvaluateCharacterDetection

import matplotlib as mpl
import matplotlib.pyplot as plt

def TestCharDetector(img, hog, rf, canon_size, alphabet, detect_idxs, save_imgs=False):
    show_times = True
    start_time = time()
    bbs = CharDetector(img, hog, rf, canon_size, alphabet, min_height=0.03,
                       detect_idxs=detect_idxs, debug=show_times, score_thr=.25)
    print 'Total char detector time: ', time() - start_time
    
    if save_imgs:
        OutputCharBbs(img, bbs, alphabet)

def ImgsToFeats(I, hog, canon_size):
    feats=np.zeros(0)

    for i in range(I.shape[3]):
        img=cv2.resize(I[:,:,:,i],canon_size,interpolation=cv.CV_INTER_CUBIC)
        feature_vector = hog.compute(img, winStride=(16,16), padding=(0,0))
        if feats.shape[0]==0:
            feats=np.zeros((feature_vector.shape[0],I.shape[3]))

        feature_vector_3d=ReshapeHog(feature_vector, (img.shape[0],img.shape[1]),
                                     hog.blockSize, hog.winSize, hog.nbins)
        feats[:,i]=feature_vector_3d.flatten()
    return feats

def TrainCharClassifier(alphabet, train_dir, bg_dir, hog, canon_size,
                        clf_path='char_clf.dat', max_per_class=np.inf,
                        max_bg=np.inf, force=False):
    random.seed(0)
    if os.path.exists(clf_path) and not(force):
        print 'Found pre-trained classifier..'
        # load classifier
        tLoad = time()
        with open(clf_path,'rb') as fid:
            rf = cPickle.load(fid)
        time_load=time()-tLoad
        print 'Character classifier loaded: ', time_load
    else:
        print 'Training new classifier...'
        # train
        # read images
        (imgs_train,y_train)=ReadAllImages(train_dir, bg_dir, alphabet,
                                           max_per_class=max_per_class, max_bg=max_bg)
        # extract features
        X_train=np.transpose(ImgsToFeats(imgs_train, hog, canon_size)).astype(np.double)
        y_train = y_train.astype(np.double)
        t1 = time()
        # NOTE: n_estimators=100 and 'entropy' gives 60% accuracy
        rf = RandomForestClassifier(n_estimators=20, n_jobs=1)
        #rf = ExtraTreesClassifier(n_estimators=100, max_depth=8,n_jobs=1, criterion='entropy')
        rf.fit(X_train, y_train)
        tTrain = time()
        time_train=tTrain-t1
        with open(clf_path,'wb') as fid:
            cPickle.dump(rf,fid)
        print 'Character classifier trained and loaded.', time_train
    return rf

def TestCharClassifier(alphabet, test_dir, bg_dir, hog, canon_size, rf,
                       max_per_class=np.inf, max_bg=np.inf):
    random.seed(0)
    (imgs_test,y_test)=ReadAllImages(test_dir, bg_dir, alphabet,
                                     max_per_class=max_per_class, max_bg=max_bg)
    X_test=np.transpose(ImgsToFeats(imgs_test, hog, canon_size)).astype(np.double)
    y_test = y_test.astype(np.double)

    t2 = time()
    score = rf.score(X_test, y_test)
    pb = rf.predict_proba(X_test)
    tTest = time()
    time_test = tTest-t2
    print 'Testing...' 
    print 'test time: ', time_test
    print "Accuracy: %0.2f" % (score)

def Bootstrap(bs_img_dir, bs_out_dir, bg_dir, canon_size,
              alphabet, detect_idxs, rf, max_per_image = 100, num_procs=1):

    # clear out previous bootstrap data
    bs_parent_dir,child = os.path.split(bs_out_dir)
    if os.path.isdir(bs_parent_dir):
        shutil.rmtree(bs_parent_dir)

    os.makedirs(bs_parent_dir)
    # copy initial BGs
    shutil.copytree(bg_dir,bs_out_dir)

    # set up temporary output dir
    temp_dir = tempfile.mkdtemp()
        
    # set up params
    min_height = 0.05
    score_thr = 0.25

    # call batch chardetector
    CharDetectorBatch(bs_img_dir, temp_dir, rf, canon_size, alphabet,
                      detect_idxs=detect_idxs, min_height=min_height,
                      score_thr=score_thr, num_procs=num_procs)

    files = os.listdir(bs_out_dir)
    files.sort()
    last_file,ext = os.path.splitext(files[-1])
    start_offset = int(last_file[1::]) + 1
    counter = 0

    # walk the img dir and check if temp_dir has a npy file
    for name in os.listdir(bs_img_dir):
        p1,ext=os.path.splitext(name)
        if ext!='.jpg':
            continue
    
        # check if precomp file exists
        char_file = os.path.join(temp_dir, name + '.char')
        if not os.path.exists(char_file):
            print "no results from: ", name
            continue

        img = cv2.imread(os.path.join(bs_img_dir,name))
        with open(char_file,'rb') as fid:
            bbs = cPickle.load(fid)
    
        # assume bbs sorted
        print "Found %d in %s" % (bbs.shape[0], name)
        for i in range(min(bbs.shape[0], max_per_image)):
            bb = bbs[i,:]
            # crop bb out of image
            patch = img[bb[0]:bb[0]+bb[2],bb[1]:bb[1]+bb[3],:]
            # save
            new_fname = "I%05i.png" % (start_offset + counter)
            full_fname = os.path.join(bs_out_dir, new_fname)
            patch_100x100=cv2.resize(patch,(100,100),interpolation=cv.CV_INTER_CUBIC)
            res = cv2.imwrite(full_fname, patch_100x100)
            counter += 1

    # clean up
    shutil.rmtree(temp_dir)

def BenchmarkCharacterDetection(rf=None):
    """
    Run character detection
    Produce graph of P/R for each character, and print avg P/R
    """

    img_dir = settings.img_test_dir
    gt_dir = settings.img_char_test_gt_dir

    temp_char_dir = 'cache/chars'

    if rf is not None:
        # batch character detection
        CharDetectorBatch(img_dir, temp_char_dir, rf, settings.canon_size,
                          settings.alphabet_master, detect_idxs=settings.detect_idxs,
                          min_height=settings.min_height,
                          min_pixel_height=settings.min_pixel_height,
                          score_thr=settings.score_thr, num_procs=settings.n_procs,
                          case_mapping=settings.case_mapping,
                          overlap_thr=settings.overlap_thr)
    else:
        print 'No classifier given, just producing results from an old run.'

    eval_results = EvaluateCharacterDetection(gt_dir, temp_char_dir)

    gt_results = eval_results[0]
    dt_results = eval_results[1]
    precision = eval_results[2]
    recall = eval_results[3]
    thrs = eval_results[4]

    # plot PR
    n_scatter=50
    fig = plt.figure()
    recall_range = np.max(recall) - np.min(recall)
    recall_step = recall_range / n_scatter

    recall_sc = np.array([recall[0]])
    precision_sc = np.array([precision[0]])
    thrs_sc = np.array([thrs[0]])
    last_recall = recall_sc[-1]
    for i in range(precision.shape[0]):
        if recall[i] > (last_recall + recall_step):
            recall_sc = np.append(recall_sc,recall[i])
            precision_sc = np.append(precision_sc,precision[i])
            thrs_sc = np.append(thrs_sc,thrs[i])
            last_recall = recall[i]
        
    plt.clf()
    plt.scatter(recall_sc, precision_sc, c=thrs_sc,marker=',',edgecolor='none')
    plt.colorbar()
    plt.hold(True)
    plt.plot(recall, precision,'g')
    plt.axis((0,1,0,1))
    plt.savefig(os.path.join(settings.fig_dir,"pr_char_curves.pdf"))

    
def main():
    # 1. train a base classifier
    rf = TrainCharClassifier(settings.alphabet_master,
                             settings.char_train_dir,
                             settings.char_train_bg_dir,
                             settings.hog,
                             settings.canon_size,
                             clf_path=settings.initial_char_clf_name,
                             max_per_class=settings.max_per_class,
                             max_bg=settings.max_bg,
                             force=True)

    # 2. 62-way
    TestCharClassifier(settings.alphabet_detect,
                       settings.char_test_dir,
                       settings.char_test_bg_dir,
                       settings.hog,
                       settings.canon_size,
                       rf)

    # 3. bootstrap
    Bootstrap(settings.bootstrap_img_dir,
              settings.bootstrap_bg_dir,              
              settings.char_train_bg_dir,
              settings.canon_size,
              settings.alphabet_master,
              settings.detect_idxs,
              rf,
              num_procs=settings.n_procs)

    # 4. train a classifier after bootstrap
    rf = TrainCharClassifier(settings.alphabet_master,
                             settings.char_train_dir,
                             settings.bootstrap_bg_dir,
                             settings.hog,
                             settings.canon_size,
                             clf_path=settings.char_clf_name,
                             max_per_class=settings.max_per_class,
                             max_bg=settings.max_bg_after_bs,
                             force=True)

    # 5. evaluate character detection
    #BenchmarkCharacterDetection(rf);
    """
    BenchmarkCharacterDetection();
    """

if __name__=="__main__":
    #cProfile.run('main()','profile_detection')
    main()
    
