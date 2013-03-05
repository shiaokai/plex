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


def TestCharDetector(img, hog, rf, canon_size, alphabet, save_imgs=False):
    '''
    Try to call RF just once to see if its any faster
    '''
    # loop over scales
    scales = [.7, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    show_times = True

    start_time = time()
    bbs = CharDetector(img, hog, rf, canon_size, alphabet, scales, debug=show_times)
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

def ReadAllImages(base_dir,char_classes,max_per_class):#,bg_dir,max_for_bg):
    # implement 'n'
    # walk directory and load in n images
    imgs=np.zeros(0)
    labels=np.zeros(0)
    k = 0
    max_allocate=5e4
    for class_index in range(len(char_classes)):
        cur_class = char_classes[class_index]
        if cur_class.islower():
            # lowercase have minus
            cur_class='-'+cur_class
        imgs_dir = os.path.join(base_dir,cur_class)
        for root, dirs, files in os.walk(imgs_dir):
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

    print 'Loaded %i images', k
    imgs=imgs[:,:,:,0:k]
    return (imgs,labels)

def TrainCharClassifier(alphabet, hog, canon_size):
    print 'Training...'
    char_clf_path='cache_char_clf.dat'
    if os.path.exists(char_clf_path):
        # load classifier
        tLoad = time()
        with open(char_clf_path,'rb') as fid:
            rf = cPickle.load(fid)
        time_load=time()-tLoad
        print 'character classifier loaded: ', time_load
    else:
        # train
        # read images
        (imgs_train,y_train)=ReadAllImages('/data/text/plex/icdar/train/charHard',alphabet,np.inf)

        # extract features
        X_train=np.transpose(ImgsToFeats(imgs_train, hog, canon_size)).astype(np.double)
        y_train = y_train.astype(np.double)
        t1 = time()
        # NOTE: n_estimators=100 and 'entropy' gives 60% accuracy
        rf = RandomForestClassifier(n_estimators=20, n_jobs=4)

        rf.fit(X_train, y_train)
        tTrain = time()
        time_train=tTrain-t1
        with open(char_clf_path,'wb') as fid:
            cPickle.dump(rf,fid)
        print 'character classifier trained and loaded.', time_train
    return rf

def TestCharClassifier(alphabet, hog, rf, canon_size):
    (imgs_test,y_test)=ReadAllImages('/data/text/plex/icdar/test/charHard',alphabet,np.inf)
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


def main():
    alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    hog = cv2.HOGDescriptor((16,16),(16,16),(16,16),(8,8),9,1,-1)
    canon_size=(48,48)
    
    # 1. train classifier
    rf=TrainCharClassifier(alphabet, hog, canon_size)
    #TestCharClassifier(alphabet, hog, rf, canon_size)

    # 2. extract full image features
    #    - a. extract features from whole image, then slice up features into groups
    #    - b. [try this first] slice up image then extract features from each slice
    #img = cv2.imread('data/IMG_2532_double.JPG')
    #img = cv2.imread('data/test_800x600.JPG')
    img = cv2.imread('data/test_1600x1200.JPG')
    TestCharDetector(img, hog, rf, canon_size, alphabet, save_imgs=False)
    
if __name__=="__main__":
    cProfile.run('main()','profile_detection')
    
