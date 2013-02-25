import pdb
import os
import random
import numpy as np
import cv,cv2

from matplotlib.pyplot import show, imshow, figure, title
from hog_utils import draw_hog

from time import time
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import fetch_mldata
from numpy import arange

def ImgsToFeats(I):
    feats=np.zeros(0)
    hog = cv2.HOGDescriptor((16,16),(16,16),(16,16),(8,8),9,1,-1)
    for i in range(I.shape[3]):
        img=I[:,:,:,i]
        feature_vector = hog.compute(img, winStride=(16,16), padding=(0,0))
        if feats.shape[0]==0:
            feats=np.zeros((feature_vector.shape[0],I.shape[3]))
        feats[:,i]=feature_vector[:,0]
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

    print 'Final k = %i', k
    imgs=imgs[:,:,:,0:k]
    return (imgs,labels)


alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# read images
(imgs_train,y_train)=ReadAllImages('/data/text/plex/icdar/train/charHard',alphabet,np.inf)
(imgs_test,y_test)=ReadAllImages('/data/text/plex/icdar/test/charHard',alphabet,np.inf)
# extract features
X_train=np.transpose(ImgsToFeats(imgs_train)).astype(np.double)
y_train = y_train.astype(np.double)
X_test=np.transpose(ImgsToFeats(imgs_test)).astype(np.double)
y_test = y_test.astype(np.double)

# 3. train
# Define training and testing sets
t1 = time()
rf = RandomForestClassifier(n_estimators=50)
# n_estimators=100 and 'entropy' gives 60% accuracy
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, y_train)
tTrain = time()
time_train=tTrain-t1

t2 = time()
score = rf.score(X_test, y_test)
pb = rf.predict_proba(X_test)
tTest = time()
time_test = tTest-t2
print 'train time: ', time_train
print 'test time: ', time_test
print "Accuracy: %0.2f" % (score)
