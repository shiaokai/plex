import settings

import pdb
import os
import random
import numpy as np
import cv,cv2
import cPickle
import cProfile

import matplotlib as mpl
import matplotlib.pyplot as plt

from hog_utils import draw_hog, ReshapeHog
from char_det import CharDetector

from time import time
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import fetch_mldata
from numpy import arange
import shutil


def OutputCharBbs(I, bbs, alphabet, output_dir='dbg'):
    Irgb = np.copy(I)
    Irgb = Irgb[:,:,[2,1,0]]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(len(alphabet)):
        DrawCharBbs(Irgb, bbs, alphabet, filter_label=i)
        # store image
        plt.savefig(os.path.join(output_dir,"charDet_%s.png" % (alphabet[i])))

def DrawCharBbs(I, bbs, alphabet, filter_label=-1, draw_top=-1):
    fig = plt.figure()
    plt.cla()
    ax = fig.add_subplot(111)
    plt.imshow(I);

    if draw_top>0:
        # sort by score
        sidx = np.argsort(bbs[:,4])
        sidx = sidx[::-1]
        for i in range(draw_top):
            if i > bbs.shape[0]:
                break
            else:
                bb = bbs[sidx[i],:]
                patch = mpl.patches.Rectangle((bb[1],bb[0]),
                                              bb[2],bb[3],
                                              color='green',
                                              fill=False)
                # draw rectangle
                ax.add_patch(patch)
                plt.text(bb[1],bb[0],"%s:%.02f" % (alphabet[int(bb[5])],float(bb[4])),
                         backgroundcolor=(1,1,1))
                # draw text
                
    else:
        for i in range(bbs.shape[0]):
            bb = bbs[i,:]
            if filter_label>=0 and bb[5] != filter_label:
                continue
            else:
                # draw me
                patch = mpl.patches.Rectangle((bb[1],bb[0]),
                                              bb[2],bb[3],
                                              color='green',
                                              fill=False)
                ax.add_patch(patch)
                plt.text(bb[1],bb[0],"%s:%.02f" % (alphabet[int(bb[5])],float(bb[4])),
                         backgroundcolor=(1,1,1))


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
    # implement 'n'
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
                        clf_path='char_clf.dat', max_per_class=np.inf, max_bg=np.inf):
    random.seed(0)
    if os.path.exists(clf_path):
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
        rf = RandomForestClassifier(n_estimators=20, n_jobs=4)
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
    scales = [.8, .9, 1.0, 1.1, 1.2]
    offset = 1
    for root, dirs, files in os.walk(bs_img_dir):
        for name in files:
            p1,ext=os.path.splitext(name)
            if ext!='.jpg':
                continue
            img = cv2.imread(os.path.join(bs_img_dir,name))
            show_times = True
            start_time = time()
            bbs = CharDetector(img, hog, rf, canon_size, alphabet, scales,
                               detect_idxs=detect_idxs, debug=show_times,
                               min_height=.05, score_thr=.05)
            # assume bbs sorted
            counter = 0
            for i in range(min(bbs.shape[0],max_per_image)):
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
    '''
    rf=TrainCharClassifier(settings.alphabet_master,
                           settings.char_train_dir,
                           settings.char_train_bg_dir,
                           settings.hog,
                           settings.canon_size)

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
              settings.hog,
              settings.canon_size,
              settings.alphabet_master,
              settings.detect_idxs,
              rf)
    '''
    # 4. train a classifier after bootstrap
    rf=TrainCharClassifier(settings.alphabet_master,
                           settings.char_train_dir,
                           settings.bootstrap_bg_dir,
                           settings.hog,
                           settings.canon_size,
                           settings.char_clf_name,
                           settings.max_per_class,
                           settings.max_bg)

    # 2. extract full image features
    #    - a. extract features from whole image, then slice up features into groups
    #    - b. [try this first] slice up image then extract features from each slice
    #img = cv2.imread('data/IMG_2532_double.JPG')
    #img = cv2.imread('data/IMG_2532.JPG')
    #img = cv2.imread('data/scales.JPG')
    #img = cv2.imread('data/test_800x600.JPG')
    img = cv2.imread('data/test_1600x1200.JPG')
    TestCharDetector(img, settings.hog, rf, settings.canon_size,
                     settings.alphabet_master, settings.detect_idxs, save_imgs=False)
    
if __name__=="__main__":
    cProfile.run('main()','profile_detection')
    
