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


def BbsNms(bbs, overlap_thr = 0, separate = True):
    '''
    NMS over bounding box list
    '''

    # sort bbs by score
    sidx = np.argsort(bbs[:,4])
    sidx = sidx[::-1]
    bbs = bbs[sidx,:]

    keep = [True]* bbs.shape[0]
    bbs_areas = bbs[:,2] * bbs[:,3]

    bbs_start_y = bbs[:,0]
    bbs_start_x = bbs[:,1]
    bbs_end_y = bbs[:,0] + bbs[:,2]
    bbs_end_x = bbs[:,1] + bbs[:,3]

    # start at highest scoring bb    
    for i in range(bbs.shape[0]):
        cur_class = -1
        if not(keep[i]):
            continue
        if separate:
            cur_class = bbs[i,5]
        for jj in range(i+1, bbs.shape[0]):
            if not(keep[jj]):
                continue
            # if separate, do nothing when classes are not equal
            if separate:
                if cur_class != bbs[jj,5]:
                    continue
            # mask out all worst scoring overlapping
            intersect_width = min(bbs_end_x[i], bbs_end_x[jj]) - max(bbs_start_x[i], bbs_start_x[jj])
            if intersect_width <= 0:
                continue
            intersect_height = min(bbs_end_y[i], bbs_end_y[jj]) - max(bbs_start_y[i], bbs_start_y[jj])
            if intersect_width <= 0:
                continue
            intersect_area = intersect_width * intersect_height
            union_area = bbs_areas[i] + bbs_areas[jj] - intersect_area
            overlap = intersect_area / union_area
            # threshold and reject
            if overlap > overlap_thr:
                keep[jj] = False

    keep_idxs=[]
    for i in range(len(keep)):
        if keep[i]:
            keep_idxs = keep_idxs + [i]
    return bbs[keep_idxs,:]

def HogResponseNms(responses, dim, score_thr = .25, olap_thr=.5):
    '''
    NMS over hog response surfaces.
    NOTE: everything gets scaled up by 8 pix
    '''
    bbs = np.zeros(0)
    # compute NMS over each class separately
    for i in range(responses.shape[2]):
        # find highest response
        # zero out all nearby responses
        cur_response = responses[:,:,i]
        cur_max = cur_response.max()
        cur_xy = np.unravel_index(cur_response.argmax(),
                                  cur_response.shape)
        while cur_max > score_thr:
            # add current guy to result bb list
            bb = np.array([cur_xy[0],
                           cur_xy[1],
                           dim[0], dim[1], cur_max, i])
            if bbs.shape[0] == 0:
                bbs = bb
            else:
                bbs = np.vstack((bbs, bb))

            '''
            i1_mask = max(0,bb[0])
            i2_mask = min(cur_response.shape[0],bb[0]+bb[2])
            j1_mask = max(0,bb[1])
            j2_mask = min(cur_response.shape[1],bb[1]+bb[3])
            '''
            
            i1_mask = max(bb[0] - dim[0] * olap_thr, 0)
            i2_mask = min(bb[0] + dim[0] * olap_thr, cur_response.shape[0])
            j1_mask = max(bb[1] - dim[1] * olap_thr, 0)
            j2_mask = min(bb[1] + dim[1] * olap_thr, cur_response.shape[1])

            cur_response[i1_mask:i2_mask, j1_mask:j2_mask]=-1
            cur_max = cur_response.max()
            cur_xy = np.unravel_index(cur_response.argmax(),
                                      cur_response.shape)

    # bring bbs back to image space: 1 cell represents 8 pixels
    for i in range(bbs.shape[0]):
        bbs[i,0] = bbs[i,0] * 8
        bbs[i,1] = bbs[i,1] * 8
        bbs[i,2] = bbs[i,2] * 8
        bbs[i,3] = bbs[i,3] * 8       

    return bbs

def TestCharDetector(img, hog, rf, canon_size, alphabet):
    # loop over scales
    scales = [1.0]
    bbs = np.zeros(0)
    t_det1 = time()    
    for scale in scales:
        new_size = (int(scale * img.shape[1]),int(scale * img.shape[0]))
        scaled_img=cv2.resize(img,new_size)
        feature_vector=hog.compute(scaled_img, winStride=(16,16), padding=(0,0))
        feature_vector_3d=ReshapeHog(scaled_img, hog, feature_vector)
        cell_height = canon_size[0]/8
        cell_width = canon_size[1]/8    
        responses = np.zeros((feature_vector_3d.shape[0]-cell_height+1,
                              feature_vector_3d.shape[1]-cell_width+1,62))
        # call the detector at each location. TODO: make more efficient
        for i in range(feature_vector_3d.shape[0]-cell_height+1):
            for j in range(feature_vector_3d.shape[1]-cell_width+1):
                feats = feature_vector_3d[i:i+cell_height,j:j+cell_width,:]
                pb = rf.predict_proba(feats.flatten()).flatten()
                if responses.shape[2]==pb.shape[0]:
                    responses[i,j,:] = pb
                else:
                    for k in range(rf.classes_.shape[0]):
                        responses[i,j,int(rf.classes_[k])] = pb[k]
                        
        # NMS over responses
        scaled_bbs = HogResponseNms(responses, (cell_height, cell_width))
        for i in range(scaled_bbs.shape[0]):
            scaled_bbs[i,0] = scaled_bbs[i,0] / scale
            scaled_bbs[i,1] = scaled_bbs[i,1] / scale
            scaled_bbs[i,2] = scaled_bbs[i,2] / scale
            scaled_bbs[i,3] = scaled_bbs[i,3] / scale                

        if bbs.shape[0]==0:
            bbs = scaled_bbs
        else:
            bbs = np.vstack((bbs,scaled_bbs))

    time_det = time() - t_det1
    print "Detection time: ", time_det 

    # NMS over bbs across scales
    bbs = BbsNms(bbs)
    OutputCharBbs(img, bbs, alphabet)

def TestCharDetector2(img, hog, rf, canon_size, alphabet):
    '''
    Try to call RF just once to see if its any faster
    '''
    # loop over scales
    scales = [.7, .8, .9, 1.0, 1.1, 1.2]
    bbs = np.zeros(0)
    t_det1 = time()    
    for scale in scales:
        new_size = (int(scale * img.shape[1]),int(scale * img.shape[0]))
        scaled_img=cv2.resize(img,new_size)
        feature_vector=hog.compute(scaled_img, winStride=(16,16), padding=(0,0))
        feature_vector_3d=ReshapeHog(scaled_img, hog, feature_vector)
        cell_height = canon_size[0]/8
        cell_width = canon_size[1]/8    
        i_windows = feature_vector_3d.shape[0]-cell_height+1
        j_windows = feature_vector_3d.shape[1]-cell_width+1
        responses2 = np.zeros((i_windows * j_windows, len(alphabet)))
        feature_window_stack = np.zeros((i_windows * j_windows, cell_height*cell_width*9))

        # call the detector at each location. TODO: make more efficient
        for i in range(i_windows):
            for j in range(j_windows):
                feats = feature_vector_3d[i:i+cell_height,j:j+cell_width,:]
                idx = np.ravel_multi_index((i,j),(i_windows,j_windows))
                feature_window_stack[idx,:] = feats.flatten()

        pb = rf.predict_proba(feature_window_stack)
        if len(alphabet)==pb.shape[1]:
            responses2 = pb
        else:
            dumb_idxs = []
            responses2[:,rf.classes_.tolist()] = pb
                        
        responses2=responses2.reshape((i_windows, -1, len(alphabet)))
        # NMS over responses
        scaled_bbs = HogResponseNms(responses2, (cell_height, cell_width))
        for i in range(scaled_bbs.shape[0]):
            scaled_bbs[i,0] = scaled_bbs[i,0] / scale
            scaled_bbs[i,1] = scaled_bbs[i,1] / scale
            scaled_bbs[i,2] = scaled_bbs[i,2] / scale
            scaled_bbs[i,3] = scaled_bbs[i,3] / scale                

        if bbs.shape[0]==0:
            bbs = scaled_bbs
        else:
            bbs = np.vstack((bbs,scaled_bbs))

    time_det = time() - t_det1
    print "Detection time: ", time_det 

    # NMS over bbs across scales
    bbs = BbsNms(bbs)
    OutputCharBbs(img, bbs, alphabet)

def ImgsToFeats(I, hog, canon_size):
    feats=np.zeros(0)

    for i in range(I.shape[3]):
        img=cv2.resize(I[:,:,:,i],canon_size)
        feature_vector = hog.compute(img, winStride=(16,16), padding=(0,0))
        if feats.shape[0]==0:
            feats=np.zeros((feature_vector.shape[0],I.shape[3]))

        feature_vector_3d=ReshapeHog(img, hog, feature_vector)
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
        rf = RandomForestClassifier(n_estimators=20)

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
    # TestCharClassifier(alphabet, hog, rf, canon_size)

    # 2. extract full image features
    #    - a. extract features from whole image, then slice up features into groups
    #    - b. [try this first] slice up image then extract features from each slice
    #img = cv2.imread('IMG_2532_double.JPG')
    img = cv2.imread('data/test_char_det.JPG')
    TestCharDetector2(img, hog, rf, canon_size, alphabet)
    
if __name__=="__main__":
    cProfile.run('main()','profile_detection')
    
