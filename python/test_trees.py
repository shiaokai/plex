import pdb
import random
import numpy

from time import time
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import fetch_mldata
from numpy import arange

mnist = fetch_mldata('MNIST original')

# Define training and testing sets
inds = arange(len(mnist.data))
test_i = random.sample(xrange(len(inds)), int(0.75*len(inds)))
train_i = numpy.delete(inds, test_i)

X_train = mnist.data[train_i].astype(numpy.double)
y_train = mnist.target[train_i].astype(numpy.double)

X_test = mnist.data[test_i].astype(numpy.double)
y_test = mnist.target[test_i].astype(numpy.double)

t1 = time()
rf = RandomForestClassifier(n_estimators=10, n_jobs=7)
rf.fit(X_train, y_train)
print 'train done'
tTrain = time()
time_train=tTrain-t1

t2 = time()
#score = rf.score(X_test, y_test)
score = rf.score(X_test, y_test)
tTest = time()
time_test = tTest-t2
print 'train time: ', time_train
print 'test time: ', time_test
print "Accuracy: %0.2f\t s" % (score)
