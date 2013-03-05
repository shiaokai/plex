# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False

from nms import BbsNms, HogResponseNms
from hog_utils import draw_hog, ReshapeHog
import cv2
from time import time

import numpy as np
cimport numpy as np

IDTYPE = np.uint8
ctypedef np.uint8_t IDTYPE_t

DTYPE64 = np.float64
ctypedef np.float64_t DTYPE64_t

DTYPE32 = np.float32
ctypedef np.float32_t DTYPE32_t

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline float float_max(float a, float b): return a if a >= b else b
cdef inline float float_min(float a, float b): return a if a <= b else b

def CharDetector(np.ndarray[IDTYPE_t, ndim=3] img, hog, rf,
                 canon_size, alphabet, min_height=0.1, max_height=1.0,
                 step_size=np.power(2,.25), debug=False, score_thr=.25, detect_idxs=[]):

    assert max_height<=1.0
    '''
    Try to call RF just once to see if its any faster
    '''
    # loop over scales
    cdef np.ndarray[IDTYPE_t, ndim=3] scaled_img
    cdef np.ndarray[DTYPE64_t, ndim=2] bbs = np.zeros((0,0), dtype=DTYPE64)
    cdef np.ndarray[DTYPE64_t, ndim=2] scaled_bbs = np.zeros((0,0), dtype=DTYPE64)
    cdef np.ndarray[DTYPE32_t, ndim=2] feature_vector
    cdef np.ndarray[DTYPE32_t, ndim=3] feature_vector_3d
    cdef np.ndarray[DTYPE32_t, ndim=1] feats
    cdef np.ndarray[DTYPE64_t, ndim=2] responses
    cdef np.ndarray[DTYPE64_t, ndim=3] responses3d
    cdef np.ndarray[DTYPE64_t, ndim=3] responses3d_sub
    cdef np.ndarray[DTYPE64_t, ndim=2] pb
    cdef np.ndarray[DTYPE32_t, ndim=2] feature_window_stack

    cdef float scale
    cdef int img_h = img.shape[0]
    cdef int img_w = img.shape[1]
    cdef int cell_height, cell_width, i_windows, j_windows, i, j, idx
    
    if debug:
        total_hog_nms = 0
        total_rf = 0
        total_hog_rsh = 0
        total_hog_cmp = 0
        t_det1 = time()    

    start_scale = max(canon_size[0]/(max_height*img.shape[0]),
                      canon_size[1]/(max_height*img.shape[1]))
    start_scale = int(np.ceil(np.log(start_scale) / np.log(step_size)))
    end_scale = max(canon_size[0]/(min_height*img.shape[0]),
                    canon_size[1]/(min_height*img.shape[1]))
    end_scale = int(np.floor(np.log(end_scale) / np.log(step_size)))

    for scale_power in range(start_scale, end_scale):
        scale = np.power(step_size, scale_power)
        scaled_img=cv2.resize(img, (int(scale * img_w), int(scale * img_h)))

        if debug:
            t_cmp0 = time()

        feature_vector = hog.compute(scaled_img, winStride=(16,16), padding=(0,0))

        if debug:
            total_hog_cmp += time() - t_cmp0
            t_rsh0 = time()

        feature_vector_3d = ReshapeHog(feature_vector, (scaled_img.shape[0],scaled_img.shape[1]),
                                       hog.blockSize, hog.winSize, hog.nbins)
        if debug:
            total_hog_rsh += time() - t_rsh0

        cell_height = canon_size[0]/8
        cell_width = canon_size[1]/8    
        i_windows = feature_vector_3d.shape[0]-cell_height+1
        j_windows = feature_vector_3d.shape[1]-cell_width+1
        responses = np.zeros((i_windows * j_windows, len(alphabet)), dtype=DTYPE64)
        feature_window_stack = np.zeros((i_windows * j_windows,
        cell_height*cell_width*9), dtype=DTYPE32)

        # call the detector at each location. TODO: make more efficient
        for i in range(i_windows):
            for j in range(j_windows):
                feats = feature_vector_3d[i:i+cell_height,j:j+cell_width,:].flatten()
                idx = np.ravel_multi_index((i,j),(i_windows,j_windows))
                feature_window_stack[idx,:] = feats

        if debug:
            t_det0 = time()

        pb = rf.predict_proba(feature_window_stack)

        if debug:
            time_det0 = time() - t_det0
            total_rf += time_det0

        #if len(alphabet)==pb.shape[1]:
        #    responses = pb
        #else:
        responses[:,rf.classes_.tolist()] = pb
                        
        responses3d=responses.reshape((i_windows, -1, len(alphabet)))
        # NMS over responses
        if debug:
            t_nms0 = time()

        if len(detect_idxs)>0:
            responses3d_sub = np.zeros((responses3d.shape[0],
                                        responses3d.shape[1],
                                        len(detect_idxs)),
                                        dtype=DTYPE64)
            responses3d_sub = responses3d[:,:,detect_idxs]
            scaled_bbs = HogResponseNms(responses3d_sub, cell_height,
                                        cell_width, score_thr=score_thr)
        else:
            scaled_bbs = HogResponseNms(responses3d, cell_height,
                                        cell_width, score_thr=score_thr)
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
        
    bbs = BbsNms(bbs)

    if debug:
        time_nms = time() - t_nms1
        print "Bbs NMS time: ", time_nms

    return bbs
