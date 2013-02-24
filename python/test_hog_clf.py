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

def ReadImages(base_p,n):
    # implement 'n'
    # walk directory and load in n images
    files = os.walk(base_p)
    imgs=np.zeros(0)
    k = 0
    for root, dirs, files in os.walk(base_p):
        for name in files:
            p1,ext=os.path.splitext(name)
            if ext!='.png':
                continue
            I = cv2.imread(os.path.join(base_p,name))
            if imgs.shape[0]==0:
                imgs=np.zeros((I.shape[0],I.shape[1],I.shape[2],1e4),dtype=np.uint8)
            imgs[:,:,:,k]=I
            k+=1
    imgs=imgs[:,:,:,0:k]
    return imgs

# 1. define paths to classes
c1_base='/data/text/plex/synth1000/train/char/A'
c2_base='/data/text/plex/synth1000/train/char/B'
c3_base='/data/text/plex/synth1000/train/char/C'

# 2. read images
c1_imgs=ReadImages(c1_base,100)
c2_imgs=ReadImages(c2_base,100)
c3_imgs=ReadImages(c3_base,100)

# 3. extract features
c1_feats=ImgsToFeats(c1_imgs)
c2_feats=ImgsToFeats(c2_imgs)
c3_feats=ImgsToFeats(c3_imgs)

# 4. concatenate into a single (X,y)

X=np.transpose(np.hstack((c1_feats,c2_feats,c2_feats)))
y=np.vstack((1*np.ones((c1_feats.shape[1],1)),
             2*np.ones((c2_feats.shape[1],1)),
             3*np.ones((c3_feats.shape[1],1))))

# 3. train
# Define training and testing sets
inds = arange(X.shape[1])
test_i = random.sample(xrange(len(inds)), int(0.95*len(inds)))
train_i = np.delete(inds, test_i)

X_train = X[train_i].astype(np.double)
y_train = np.squeeze(y[train_i].astype(np.double))

X_test = X[test_i].astype(np.double)
y_test = np.squeeze(y[test_i].astype(np.double))
t1 = time()
rf = RandomForestClassifier(n_estimators=10, n_jobs=1)
rf.fit(X_train, y_train)
tTrain = time()
time_train=tTrain-t1

t2 = time()
#score = rf.score(X_test, y_test)
score = rf.score(X_test, y_test)
tTest = time()
time_test = tTest-t2
print 'train time: ', time_train
print 'test time: ', time_test
print "Accuracy: %0.2f" % (score)
