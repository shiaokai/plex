import pdb
import numpy as np

def HogResponseNms(responses, cell_height, cell_width, score_thr = .25, olap_thr=.5):
    '''
    NMS over hog response surfaces.
    NOTE: everything gets scaled up by 8 pix
    '''
    '''
    max_bbs_h = responses.shape[0] / (olap_thr * cell_height * 2)
    max_bbs_w = responses.shape[1] / (olap_thr * cell_width * 2)
    '''
    # compute upper bound on bbs

    max_bbs = np.sum(responses>score_thr)
    #print 'max bbs: ', max_bbs
    bbs = np.zeros((max_bbs,6))
    k = 0
    # compute NMS over each class separately
    for i in range(responses.shape[2]):
        # find highest response
        # zero out all nearby responses
        cur_response = responses[:,:,i]
        cur_max = cur_response.max()
        cur_y,cur_x = np.unravel_index(cur_response.argmax(),
                                       cur_response.shape)
        while cur_max > score_thr:
            # add current guy to result bb list
            bbs[k,0] = cur_y
            bbs[k,1] = cur_x
            bbs[k,2] = cell_height
            bbs[k,3] = cell_width
            bbs[k,4] = cur_max
            bbs[k,5] = i
            
            i1_mask = max(bbs[k,0] - cell_height * olap_thr, 0)
            i2_mask = min(bbs[k,0] + cell_height * olap_thr, cur_response.shape[0])
            j1_mask = max(bbs[k,1] - cell_width * olap_thr, 0)
            j2_mask = min(bbs[k,1] + cell_width * olap_thr, cur_response.shape[1])

            cur_response[i1_mask:i2_mask, j1_mask:j2_mask]=-1
            cur_max = cur_response.max()
            cur_y,cur_x = np.unravel_index(cur_response.argmax(),
                                           cur_response.shape)
            k+=1


    bbs = bbs[0:k,:]
    # bring bbs back to image space: 1 cell represents 8 pixels
    scale_by = np.ones((bbs.shape[0],bbs.shape[1]))
    scale_by[:,0:4] = 8
    bbs = np.multiply(bbs, scale_by)
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

def WordBbsNms(words, overlap_thr = 0):
    # words = ((wordbb, score, char_bbs))
    # create a single Bbs tuple out of words
    
    words_bbs = np.zeros((len(words),6))
    for i in range(len(words)):
        cur_word = words[i]
        words_bbs[i,0:4] = cur_word[0]
        words_bbs[i,4] = cur_word[1]
        words_bbs[i,5] = i

    words_bbs_nms = BbsNms(words_bbs, overlap_thr = overlap_thr, separate = False)
    out_words = []
    for i in range(words_bbs_nms.shape[0]):
        out_words.append(words[int(words_bbs_nms[i,5])])
        
    return out_words


    
