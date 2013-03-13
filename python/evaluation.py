import os
import pdb
import cPickle
import numpy as np
from helpers import ValidateString, BbsOverlap
from operator import itemgetter

def ComputePrecisionRecall(gt_results, dt_results):
    assert(len(gt_results)==len(dt_results))
    # count GTs
    total_positives = 0
    for gt_result in gt_results:
        total_positives += len(gt_result)
            
    dt_pairs=np.zeros((0,2))
    for dt_result in dt_results:
        for dt in dt_result:
            dt_pairs = np.vstack((dt_pairs,
                                  np.array([dt[3],dt[1]])))

    # sort dt_pairs
    dt_pairs = np.flipud(dt_pairs[dt_pairs[:,0].argsort(),:])
    
    tp_cumsum = np.cumsum(dt_pairs[:,1])
    fp_cumsum = np.cumsum(1 - dt_pairs[:,1])
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / total_positives
    thrs = dt_pairs[:,0]

    return (precision, recall, thrs)

def EvaluateWordDetection(gt_dir, dt_dir, olap_thr=.5):

    # measure precision and recall for detection
    gt_results = []
    dt_results = []
    for gt_file in os.listdir(gt_dir):
        img_name,ext=os.path.splitext(gt_file)
        if ext!='.txt':
            continue

        # read stuff from gt file
        gt_path = os.path.join(gt_dir, gt_file)
        with open(gt_path, 'r') as f:
            gt_lines = f.readlines()
        gt_list = []
        for gt_line in gt_lines:
            if gt_line[0] == '%':
                continue
            gt_parts = gt_line.split(' ')
            (val, gt_word) = ValidateString(gt_parts[0])
            if not val:
                continue
            gt_x = int(gt_parts[1])
            gt_y = int(gt_parts[2])
            gt_w = int(gt_parts[3])
            gt_h = int(gt_parts[4])            

            gt_item = [gt_word, 0, np.array([gt_y, gt_x, gt_h, gt_w]), gt_file]
            gt_list.append(gt_item)

        # read stuff from dt file
        dt_path = os.path.join(dt_dir, img_name + '.word')
        with open(dt_path,'rb') as fid:
            word_results = cPickle.load(fid)
        dt_list = []
        for word_result in word_results:
            dt_item = [word_result[2], 0, word_result[0][0,0:4], word_result[0][0,4]]
            dt_list.append(dt_item)

        # sort dt_list
        dt_list = sorted(dt_list, key=itemgetter(3))
        dt_list.reverse()

        # greedily match dts to gts
        for i in range(len(dt_list)):
            dt_item = dt_list[i]
            # check if item overlaps with any previously matched dt_item
            for j in range(0,i):
                prev_dt_item = dt_list[j]
                if not(prev_dt_item[1]) or (prev_dt_item[0] != dt_item[0]):
                    continue
                overlap = BbsOverlap(prev_dt_item[2], dt_item[2])
                if overlap > olap_thr:
                    continue
            
            # check if item overlaps with any unmatched gt_item
            for j in range(len(gt_list)):
                gt_item = gt_list[j]
                if gt_item[1] or (gt_item[0] != dt_item[0]):
                    continue
                overlap = BbsOverlap(gt_item[2], dt_item[2])
                if overlap > olap_thr:
                    gt_item[1] = 1
                    dt_item[1] = 1
                
        dt_results.append(dt_list)
        gt_results.append(gt_list)

    (precision, recall, thrs) = ComputePrecisionRecall(gt_results, dt_results)
    return (gt_results, dt_results, precision, recall, thrs)
        
    # for each ground truth word
    #   find all detected words that overlap > olap_thr
    #   associate with it the detected word of highest confidence
    #

    # return a list of objects: 
    #   (dt, gt)
    #   dt: (word, 1/0 - match, boundingbox, confidence)
    #   gt: (word, 1/0 - match, boundingbox)
