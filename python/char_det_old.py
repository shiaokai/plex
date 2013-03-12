import pdb
import os
import random
import numpy as np
import cv,cv2
import cPickle
import cProfile
import shutil
import multiprocessing as mp

import matplotlib as mpl
import matplotlib.pyplot as plt
from hog_utils_old import draw_hog, ReshapeHog
from nms import BbsNms, HogResponseNms
from time import time
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import fetch_mldata
from numpy import arange
from helpers import CollapseLetterCase

import settings

def CharDetectorBatch(img_dir, output_dir, rf, canon_size, alphabet,
                      min_height=0.1, max_height=1.0,
                      step_size=np.power(2,.25), score_thr=.25,
                      min_pixel_height = 20,
                      detect_idxs=[], debug=False,
                      case_mapping=[], num_procs=1):

    # clear output_dir
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    jobs = []
    for name in os.listdir(img_dir):
        p1,ext=os.path.splitext(name)
        if ext!='.jpg':
            continue

        save_path = os.path.join(output_dir, name + '.char')
        job = (img_dir, name, rf, canon_size, alphabet, detect_idxs,
               min_height, score_thr, save_path)
        jobs.append(job)

    if num_procs == 1:
        for job in jobs:
            CharDetectorBatchWorker(job)
    else:
        print 'using ', num_procs, ' processes to work on ', len(jobs), ' jobs.'
        pool=mp.Pool(processes=num_procs)
        pool.map_async(CharDetectorBatchWorker, jobs)
        pool.close()
        pool.join()
    
def CharDetectorBatchWorker(job):
    (img_dir, name, rf, canon_size, alphabet, detect_idxs, min_height, score_thr, save_path) = job
                   
    img = cv2.imread(os.path.join(img_dir,name))
    bbs = CharDetector(img, settings.hog, rf, canon_size, alphabet,
                       detect_idxs=detect_idxs, debug=False,
                       min_height=min_height, score_thr=score_thr)

    with open(save_path,'wb') as fid:
        cPickle.dump(bbs,fid)

    print name

#@profile
def CharDetector(img, hog, rf, canon_size, alphabet,
                 min_height=0.1, max_height=1.0,
                 step_size=np.power(2,.25), score_thr=.25,
                 min_pixel_height = 20,
                 detect_idxs=[], debug=False,
                 case_mapping=[]):


    assert max_height<=1.0

    '''
    Try to call RF just once to see if its any faster
    '''
    # loop over scales
    bbs = np.zeros(0)
    if debug:
        total_hog_nms = 0
        total_rf = 0
        total_hog_rsh = 0
        total_hog_cmp = 0
        t_det1 = time()    

    # compute scale range
    start_scale = max(canon_size[0]/(max_height*img.shape[0]),
                    canon_size[1]/(max_height*img.shape[1]))
    start_scale = int(np.ceil(np.log(start_scale) / np.log(step_size)))
    end_scale = max(canon_size[0]/(min_height*img.shape[0]),
                    canon_size[1]/(min_height*img.shape[1]))
    end_scale = int(np.floor(np.log(end_scale) / np.log(step_size)))

    cell_height = canon_size[0]/8
    cell_width = canon_size[1]/8    

    for scale_power in range(start_scale, end_scale):
        scale = np.power(step_size, scale_power)

        if (canon_size[0] / scale) < min_pixel_height:
            continue

        new_size = (int(scale * img.shape[1]),int(scale * img.shape[0]))
        scaled_img=cv2.resize(img,new_size)

        if debug:
            t_cmp0 = time()

        feature_vector=hog.compute(scaled_img, winStride=(16,16), padding=(0,0))

        if debug:
            total_hog_cmp += time() - t_cmp0
            t_rsh0 = time()

        feature_vector_3d=ReshapeHog(feature_vector, (scaled_img.shape[0],scaled_img.shape[1]),
                                     hog.blockSize, hog.winSize, hog.nbins)

        if debug:
            total_hog_rsh += time() - t_rsh0

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

        if debug:
            t_det0 = time()

        pb = rf.predict_proba(feature_window_stack)

        if debug:
            time_det0 = time() - t_det0
            total_rf += time_det0

        responses2[:,rf.classes_.tolist()] = pb
        responses2=responses2.reshape((i_windows, -1, len(alphabet)))
        # NMS over responses
        if debug:
            t_nms0 = time()

        if len(detect_idxs)>0:
            responses_subset = np.zeros((responses2.shape[0],
                                         responses2.shape[1], len(detect_idxs)))
            responses_subset = responses2[:,:,detect_idxs]
            scaled_bbs = HogResponseNms(responses_subset, cell_height, cell_width,
                                        score_thr=score_thr)
        else:
            scaled_bbs = HogResponseNms(responses2, cell_height, cell_width,
                                        score_thr=score_thr)

        if debug:
            total_hog_nms += time() - t_nms0 
        for i in range(scaled_bbs.shape[0]):
            scaled_bbs[i,0] = scaled_bbs[i,0] / scale
            scaled_bbs[i,1] = scaled_bbs[i,1] / scale
            scaled_bbs[i,2] = scaled_bbs[i,2] / scale
            scaled_bbs[i,3] = scaled_bbs[i,3] / scale                

        if bbs.shape[0]==0:
            bbs = scaled_bbs
        else:
            bbs = np.vstack((bbs,scaled_bbs))

    if debug:
        time_det = time() - t_det1
        print "Total: ", time_det 
        print "RF time: ", total_rf
        print "HOG compute time: ", total_hog_cmp
        print "HOG reshape time: ", total_hog_rsh
        print "HOG nms time: ", total_hog_nms
        # NMS over bbs across scales
        t_nms1 = time()
        
    if not(case_mapping):
        bbs = BbsNms(bbs)
    else:
        bbs = CollapseLetterCase(bbs, case_mapping)
        bbs = BbsNms(bbs)
        
    if debug:
        time_nms = time() - t_nms1
        print "Bbs NMS time: ", time_nms

    return bbs
