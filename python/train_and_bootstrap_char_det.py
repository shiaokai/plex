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

from hog_utils import draw_hog, ReshapeHog
from char_det_old import CharDetector

from time import time
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import fetch_mldata
from numpy import arange
import shutil

from display import OutputCharBbs, DrawCharBbs
from helpers import ReadAllImages

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
        img=cv2.resize(I[:,:,:,i],canon_size)
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

def BootstrapWorker(job):
    (bs_img_dir, name, rf, canon_size, alphabet, detect_idxs, min_height, score_thr, bs_out_dir, max_per_image) = job
                   
    img = cv2.imread(os.path.join(bs_img_dir,name))
    bbs = CharDetector(img, settings.hog, rf, canon_size, alphabet,
                       detect_idxs=detect_idxs, debug=False,
                       min_height=min_height, score_thr=score_thr)

    bg_result_dir = os.path.join(bs_out_dir, name)
    os.makedirs(bg_result_dir)

    if bbs.shape[0]==0:
        print "Mined none from %s" % (name)
        return

    # assume bbs sorted
    print "Mined %d from %s" % (bbs.shape[0], name)
    for i in range(min(bbs.shape[0], max_per_image)):
        bb = bbs[i,:]
        # crop bb out of image
        patch = img[bb[0]:bb[0]+bb[2],bb[1]:bb[1]+bb[3],:]
        # save
        new_fname = "I%05i.png" % (i)
        full_fname = os.path.join(bg_result_dir, new_fname)
        patch_100x100=cv2.resize(patch,(100,100))
        res = cv2.imwrite(full_fname, patch_100x100)
            
def Bootstrap_mp(bs_img_dir, bs_out_dir, bg_dir, canon_size,
                 alphabet, detect_idxs, rf, max_per_image = 100, num_procs=1):

    # clear out previous bootstrap data
    bs_parent_dir,child = os.path.split(bs_out_dir)
    if os.path.isdir(bs_parent_dir):
        shutil.rmtree(bs_parent_dir)

    os.makedirs(bs_parent_dir)
    # copy initial BGs
    shutil.copytree(bg_dir,bs_out_dir)

    jobs = []
    min_height = 0.05
    score_thr = 0.25
    for root, dirs, files in os.walk(bs_img_dir):
        for name in files:
            p1,ext=os.path.splitext(name)
            if ext!='.jpg':
                continue

            job = (bs_img_dir, name, rf,
                   canon_size, alphabet, detect_idxs,
                   min_height, score_thr, bs_out_dir, max_per_image)
            jobs.append(job)
                   
    if num_procs == 1:
        for job in jobs:
            BootstrapWorker(job)
    else:
        print 'using ', num_procs, ' processes to work on ', len(jobs), ' jobs.'
        pool=mp.Pool(processes=num_procs)
        pool.map_async(BootstrapWorker, jobs)
        pool.close()
        pool.join()

def main():
    '''
    # 1. train a base classifier
    rf=TrainCharClassifier(settings.alphabet_master,
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
    Bootstrap_mp(settings.bootstrap_img_dir,
                 settings.bootstrap_bg_dir,              
                 settings.char_train_bg_dir,
                 settings.canon_size,
                 settings.alphabet_master,
                 settings.detect_idxs,
                 rf,
                 num_procs=settings.n_procs)
    '''
    # 4. train a classifier after bootstrap
    TrainCharClassifier(settings.alphabet_master,
                        settings.char_train_dir,
                        settings.bootstrap_bg_dir,
                        settings.hog,
                        settings.canon_size,
                        clf_path=settings.char_clf_name,
                        max_per_class=settings.max_per_class,
                        max_bg=settings.max_bg,
                        force=True)

if __name__=="__main__":
    cProfile.run('main()','profile_detection')
    
