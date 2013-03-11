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

def TestCharDetector(img, hog, rf, canon_size, alphabet, detect_idxs, save_imgs=False):
    '''
    Try to call RF just once to see if its any faster
    '''
    # loop over scales
    show_times = True

    start_time = time()
    bbs = CharDetector(img, hog, rf, canon_size, alphabet, min_height=0.03,
                       detect_idxs=detect_idxs, debug=show_times, score_thr=.25)
    print 'Char detector time: ', time() - start_time
    
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


def ReadAllImages(char_dir, bg_dir, char_classes,
                  max_per_class=np.inf, max_bg=np.inf):
    '''
    This function follows up to one directory down
    '''

    # walk directory and load in n images
    imgs=np.zeros(0)
    labels=np.zeros(0)
    k = 0
    max_allocate=5e4
    for class_index in range(len(char_classes)):
        cur_class = char_classes[class_index]
        if cur_class == '_':
            imgs_dir = bg_dir  # background class
        else:
            if cur_class.islower():
                cur_class='-'+cur_class  # lowercase have minus
            imgs_dir = os.path.join(char_dir,cur_class)

        for root, dirs, files in os.walk(imgs_dir):
            # add all the files within dirs to the files list
            for cur_dir in dirs:
                for root1, dirs1, files1 in os.walk(os.path.join(imgs_dir,cur_dir)):
                    if not files1:
                        break
                    files1_with_parents = [cur_dir + os.sep + f for f in files1]
                    files += files1_with_parents
                    
            if cur_class == '_':
                if (max_bg < np.inf) and (len(files) > max_bg):
                    random.shuffle(files)
                    files = files[0:max_bg]

            else:
                if (max_per_class < np.inf) and (len(files) > max_per_class):
                    random.shuffle(files)
                    files = files[0:max_per_class]


            for name in files:
                p1,ext=os.path.splitext(name)
                if ext!='.png':
                    continue
                I = cv2.imread(os.path.join(imgs_dir,name))
                if imgs.shape[0]==0:
                    imgs=np.zeros((I.shape[0],I.shape[1],I.shape[2],max_allocate),
                                  dtype=np.uint8)
                if k<max_allocate:
                    imgs[:,:,:,k]=I
                else:
                    print 'WARNING: loading more data than max_allocate. do something!'
                    imgs=np.concatenate((imgs,I[...,np.newaxis]),axis=3)

                labels=np.append(labels,class_index)
                k+=1

    print 'Loaded %i images' % k
    imgs=imgs[:,:,:,0:k]
    return (imgs,labels)

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



def Bootstrap(bs_img_dir, bs_out_dir, bg_dir, hog, canon_size,
              alphabet, detect_idxs, rf, max_per_image = 100):

    # clear out previous bootstrap data
    bs_parent_dir,child = os.path.split(bs_out_dir)
    if os.path.isdir(bs_parent_dir):
        shutil.rmtree(bs_parent_dir)
        os.makedirs(bs_parent_dir)

    # copy initial BGs
    shutil.copytree(bg_dir,bs_out_dir)
    # figure out latest name
    start_counter = 0
    last_fname = []
    for root, dirs, files in os.walk(bs_out_dir):
        last_fname = sorted(files)
        last_fname = last_fname[-1]

    name,ext = os.path.splitext(last_fname)
    start_counter = int(name[1::])
    offset = 1
    for root, dirs, files in os.walk(bs_img_dir):
        for name in files:
            p1,ext=os.path.splitext(name)
            if ext!='.jpg':
                continue
            img = cv2.imread(os.path.join(bs_img_dir,name))
            start_time = time()
            bbs = CharDetector(img, hog, rf, canon_size, alphabet,
                               detect_idxs=detect_idxs, debug=False,
                               min_height=.05, score_thr=.25)
            # assume bbs sorted
            print "Mined %d from image: %s" % (bbs.shape[0], name)
            for i in range(min(bbs.shape[0], max_per_image)):
                bb = bbs[i,:]
                # crop bb out of image
                patch = img[bb[0]:bb[0]+bb[2],bb[1]:bb[1]+bb[3],:]
                # save
                new_fname = "I%05i.png" % (start_counter + offset)
                full_fname = os.path.join(bs_out_dir, new_fname)
                patch_100x100=cv2.resize(patch,(100,100))
                res = cv2.imwrite(full_fname, patch_100x100)
                offset += 1



def main():

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

    # 4. train a classifier after bootstrap
    rf=TrainCharClassifier(settings.alphabet_master,
                           settings.char_train_dir,
                           settings.bootstrap_bg_dir,
                           settings.hog,
                           settings.canon_size,
                           clf_path=settings.char_clf_name,
                           max_per_class=settings.max_per_class,
                           max_bg=settings.max_bg,
                           force=True)

    # 2. extract full image features
    #    - a. extract features from whole image, then slice up features into groups
    #    - b. [try this first] slice up image then extract features from each slice
    #img = cv2.imread('data/IMG_2532_double.JPG')
    #img = cv2.imread('data/IMG_2532.JPG')
    #img = cv2.imread('data/scales.JPG')
    #img = cv2.imread('data/test_800x600.JPG')
    #img = cv2.imread('data/test_1600x1200.JPG')
    #TestCharDetector(img, settings.hog, rf, settings.canon_size,
    #                 settings.alphabet_master, settings.detect_idxs, save_imgs=True)
    
if __name__=="__main__":
    cProfile.run('main()','profile_detection')
    
