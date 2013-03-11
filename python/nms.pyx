# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline float float_max(float a, float b): return a if a >= b else b
cdef inline float float_min(float a, float b): return a if a <= b else b

def HogResponseNms(np.ndarray[DTYPE_t, ndim=3] responses not None,
                   float cell_height, float cell_width,
                   float score_thr =.25, float olap_thr=.5):
    '''
    NMS over hog response surfaces.
    NOTE: everything gets scaled up by 8 pix
    '''

    cdef int max_bbs = np.sum(responses>score_thr)
    cdef int k = 0
    # print 'max bbs: ', max_bbs
    # this might crash if too big?
    cdef np.ndarray[DTYPE_t, ndim=2] bbs = np.zeros((max_bbs,6), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] scale_by
    cdef np.ndarray[DTYPE_t, ndim=2] cur_response
    cdef DTYPE_t cur_max
    cdef int cur_x, cur_y
    cdef int r_height = responses.shape[0]
    cdef int r_width = responses.shape[1]
    cdef int i
    cdef int i1_mask, i2_mask, j1_mask, j2_mask
    cdef int mask_height = <int>(cell_height * olap_thr)
    cdef int mask_width = <int>(cell_width * olap_thr)


    # compute NMS over each class separately
    for i in range(responses.shape[2]):
        # find highest response
        # zero out all nearby responses
        cur_response = responses[:,:,i]
        cur_max = cur_response.max()
        cur_y, cur_x = np.unravel_index(cur_response.argmax(),(r_height,r_width))
        while cur_max > score_thr:
            # add current guy to result bb list
            bbs[k,0] = cur_y
            bbs[k,1] = cur_x
            bbs[k,2] = cell_height
            bbs[k,3] = cell_width
            bbs[k,4] = cur_max
            bbs[k,5] = i

            i1_mask = int_max((<int>bbs[k,0]) - mask_height, 0)
            i2_mask = int_min((<int>bbs[k,0]) + mask_height, r_height)
            j1_mask = int_max((<int>bbs[k,1]) - mask_width, 0)
            j2_mask = int_min((<int>bbs[k,1]) + mask_width, r_width)

            cur_response[i1_mask:i2_mask, j1_mask:j2_mask]=-1
            cur_max = cur_response.max()
            cur_y, cur_x = np.unravel_index(cur_response.argmax(), (r_height,r_width))
            k+=1

    bbs = bbs[0:k,:]
    # bring bbs back to image space: 1 cell represents 8 pixels
    scale_by = np.ones((bbs.shape[0],bbs.shape[1]), dtype=DTYPE)
    scale_by[:,0:4] = 8
    bbs = np.multiply(bbs, scale_by)
    return bbs


def BbsNms(np.ndarray[DTYPE_t, ndim=2] bbs, overlap_thr = 0, separate = True):
    '''
    NMS over bounding box list
    '''

    if bbs.shape[0] == 0:
       return np.zeros((0,6), dtype=DTYPE)

    # sort bbs by score
    cdef np.ndarray[np.int_t, ndim=1] sidx = np.argsort(bbs[:,4])
    sidx = sidx[::-1]
    bbs = bbs[sidx,:]

    keep = [True] * bbs.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] bbs_areas = bbs[:,2] * bbs[:,3]
    cdef np.ndarray[DTYPE_t, ndim=1] bbs_start_y = bbs[:,0]
    cdef np.ndarray[DTYPE_t, ndim=1] bbs_start_x = bbs[:,1]
    cdef np.ndarray[DTYPE_t, ndim=1] bbs_end_y = bbs[:,0] + bbs[:,2]
    cdef np.ndarray[DTYPE_t, ndim=1] bbs_end_x = bbs[:,1] + bbs[:,3]
    cdef DTYPE_t cur_class
    cdef int i, jj
    cdef DTYPE_t intersect_width, intersect_height
    cdef DTYPE_t intersect_area, union_area
    cdef DTYPE_t overlap
    cdef DTYPE_t bbs_end_x_i, bbs_end_y_i, bbs_areas_i
    cdef DTYPE_t bbs_start_x_i, bbs_start_y_i
    
    # start at highest scoring bb    
    for i in range(bbs.shape[0]):
        cur_class = -1
        if not(keep[i]):
            continue
        if separate:
            cur_class = bbs[i,5]
        bbs_end_x_i = bbs_end_x[i]
        bbs_end_y_i = bbs_end_y[i]
        bbs_areas_i = bbs_areas[i]
        bbs_start_x_i = bbs_start_x[i]
        bbs_start_y_i = bbs_start_y[i]
        for jj in range(i+1, bbs.shape[0]):
            if not(keep[jj]):
                continue
            # if separate, do nothing when classes are not equal
            if separate:
                if int(cur_class) != int(bbs[jj,5]):
                    continue
            # mask out all worst scoring overlapping
            intersect_width = float_min(bbs_end_x_i, bbs_end_x[jj]) - float_max(bbs_start_x_i, bbs_start_x[jj])
            if intersect_width <= 0:
                continue
            intersect_height = float_min(bbs_end_y_i, bbs_end_y[jj]) - float_max(bbs_start_y_i, bbs_start_y[jj])
            if intersect_width <= 0:
                continue
            intersect_area = intersect_width * intersect_height
            union_area = bbs_areas_i + bbs_areas[jj] - intersect_area
            overlap = intersect_area / union_area
            # threshold and reject
            if overlap > overlap_thr:
                keep[jj] = False

    keep_idxs=[]
    for i in range(len(keep)):
        if keep[i]:
            keep_idxs = keep_idxs + [i]
    return bbs[keep_idxs,:]


def WordBbsNms(words, overlap_thr = 0):
    # words = ((wordbb, score, char_bbs))
    # create a single Bbs tuple out of words
    
    cdef np.ndarray[DTYPE_t, ndim=2] words_bbs = np.zeros((len(words),6), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] words_bbs_nms
    
    for i in range(len(words)):
        cur_word = words[i]
        words_bbs[i,0:4] = cur_word[0]
        words_bbs[i,4] = -cur_word[1]
        words_bbs[i,5] = i

    words_bbs_nms = BbsNms(words_bbs, overlap_thr = overlap_thr, separate = False)
    out_words = []
    for i in range(words_bbs_nms.shape[0]):
        out_words.append(words[int(words_bbs_nms[i,5])])
        
    return out_words
    
