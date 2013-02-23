import pdb
import os
import random
import numpy

from time import time
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import fetch_mldata
from numpy import arange

def readImages(base_p,n):
    # walk directory and load in n images
    files = os.walk(base_p)
    for root, dirs, files in os.walk(base_p):
        for name in files:
            pdb.set_trace()
            p1,ext=os.path.splitext(files)
            if ext!='png':
                continue
        

# 1. read images: 3 classes for now
c1_base='/data/text/plex/synth1000/train/char/A'
c1_imgs=readImages(c1_base,100)
c2_base='/data/text/plex/synth1000/train/char/B'
c2_imgs=readImages(c2_base,100)
c3_base='/data/text/plex/synth1000/train/char/C'
c3_imgs=readImages(c3_base,100)
# 2. extract hog features

# 3. train

mnist = fetch_mldata('MNIST original')

# Define training and testing sets
inds = arange(len(mnist.data))
test_i = random.sample(xrange(len(inds)), int(0.1*len(inds)))
train_i = numpy.delete(inds, test_i)

X_train = mnist.data[train_i].astype(numpy.double)
y_train = mnist.target[train_i].astype(numpy.double)

X_test = mnist.data[test_i].astype(numpy.double)
y_test = mnist.target[test_i].astype(numpy.double)

t1 = time()
rf = RandomForestClassifier(n_estimators=10, n_jobs=1)
rf.fit(X_train, y_train)
tTrain = time()
time_train=tTrain-t1

t2 = time()
#score = rf.score(X_test, y_test)
score = rf.score(X_test, y_test)
pdb.set_trace()
tTest = time()
time_test = tTest-t2
print 'train time: ', time_train
print 'test time: ', time_test
print "Accuracy: %0.2f\t%0.2f s" % (score, dt)
