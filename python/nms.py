import numpy as np
from time import time

def HogResponseNms(responses, dim, score_thr = .25, olap_thr=.5):
    t1 = time()
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

    time_nms = time() - t1
    print "Response NMS time: ", time_nms
    return bbs

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
