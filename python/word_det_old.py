import settings

import pdb
import os
import random
import numpy as np
import cv,cv2
import cPickle
import cProfile

import matplotlib as mpl
import matplotlib.pyplot as plt
from nms import BbsNms, HogResponseNms
from time import time

# fix bug
def UnionBbs(bbs):
    right = -1
    bottom = -1
    left = np.inf
    top = np.inf
    for i in range(bbs.shape[0]):
        if bbs[i,0] < top:
            top = bbs[i,0]
        if bbs[i,1] < left:
            left = bbs[i,1]
        if (bbs[i,0] + bbs[i,2]) > bottom:
            bottom = bbs[i,0] + bbs[i,2]
        if (bbs[i,1] + bbs[i,3]) > right:
            right = bbs[i,1] + bbs[i,3]
                
    u_bb = np.array([top, left, bottom - top, right - left])
    return u_bb

def WordDetector(bbs, lexicon, alphabet):
    results = []
    for i in range(len(lexicon)):
        word = lexicon[i]
        # assume word is at least 3 characters long
        assert len(word) > 2
        (word_bb, word_score, best_bbs) = SolveWord(bbs, word, alphabet)
        word_result = np.append(word_bb, [word_score, 0])
        results.append((np.expand_dims(word_result, axis = 0), best_bbs))

    return results

def SolveWord(bbs, word, alphabet, alpha = .5):
    # store costs and pointers
    dp_costs = []
    dp_ptrs = []

    process_order = range(len(word)-1,-1,-1)
    # iterate from the leaf to the node before the root
    # use 'i' as an increasing index, but refer to
    #   process_order for the DP
    for i in range(len(word) - 1):
        i_proc = process_order[i]
        child_char = word[i_proc]
        child_idx  = alphabet.find(child_char)
        child_bbs  = bbs[bbs[:,5]==child_idx,:]

        parent_char = word[i_proc-1]
        parent_idx  = alphabet.find(parent_char)
        parent_bbs  = bbs[bbs[:,5]==parent_idx,:]

        num_parents = parent_bbs.shape[0]
        num_children = child_bbs.shape[0]

        dp_costs_j = -1 * np.ones(num_parents)
        dp_ptrs_j = -1 * np.ones(num_parents)

        if (i == 0):
            # process leaf nodes
            for j in range(num_parents):
                parent_bb = parent_bbs[j,:]
                best_child_score = np.inf
                best_child_idx = -1
                for k in range(num_children):
                    child_bb = child_bbs[k]
                    # compute pairwise score
                    score = ComputePairScore(parent_bb, child_bb)
                    if best_child_score > score:
                        best_child_score = score
                        best_child_idx = k

                # store min score and index
                dp_costs_j[j] = best_child_score
                dp_ptrs_j[j] = best_child_idx

            # append to back (but later reverse)
            dp_costs.append(dp_costs_j)
            dp_ptrs.append(dp_ptrs_j)
        else:
            # process internal nodes
            dp_costs_child = dp_costs[i-1]
            for j in range(num_parents):
                parent_bb = parent_bbs[j,:]
                best_child_score = np.inf
                best_child_idx = -1
                for k in range(num_children):
                    child_bb = child_bbs[k]
                    # compute pairwise score
                    score = ComputePairScore(parent_bb, child_bb) + dp_costs_child[k]
                    if best_child_score > score:
                        best_child_score = score
                        best_child_idx = k
                # store min score and index
                dp_costs_j[j] = best_child_score
                dp_ptrs_j[j] = best_child_idx
            # append to back (but later reverse)
            dp_costs.append(dp_costs_j)
            dp_ptrs.append(dp_ptrs_j)

    # process root
    child_char = word[1]
    child_idx  = alphabet.find(child_char)
    child_bbs  = bbs[bbs[:,5]==child_idx,:]

    root_char = word[0]
    root_idx  = alphabet.find(root_char)
    root_bbs  = bbs[bbs[:,5]==root_idx,:]

    num_roots = root_bbs.shape[0]
    num_children = child_bbs.shape[0]
    
    dp_costs_j = -1 * np.ones(num_roots)
    dp_ptrs_j = -1 * np.ones(num_roots)

    dp_costs_root = dp_costs[-1]
            
    best_root_score = np.inf
    best_root_idx = -1

    for j in range(num_roots):
        root_bb = root_bbs[j,:]
        score = (1 - root_bb[4]) * (1-alpha) + dp_costs_root[j]
        if best_root_score > score:
            best_root_score = score
            best_root_idx = j

    if best_root_score == np.inf:
        print "Best match is infinite?"
        assert 0

    # collect results from optimal root
    best_bbs_idx = [best_root_idx]
    dp_ptrs.reverse()
    for i in range(len(word)-1):
        dp_ptrs_j = dp_ptrs[i]
        best_bbs_idx.append(dp_ptrs_j[best_bbs_idx[i]])
    
    best_bbs = np.zeros((len(word),6))
    for i in range(len(word)):
        bb_idx = best_bbs_idx[i]
        char_idx  = alphabet.find(word[i])
        char_bbs = bbs[bbs[:,5]==char_idx]
        if (char_bbs.ndim == 1):
            best_bbs[i,:] = char_bbs
        else:
            best_bbs[i,:] = char_bbs[bb_idx,:]

    # TODO: return top K not just 1
    word_bb = UnionBbs(best_bbs)
    return (word_bb, best_root_score, best_bbs)
    
def ComputePairScore(parent_bb, child_bb, alpha = .5):    
    # TODO: if child is to left of parent, return inf cost
    
    # costs of x and y offsets
    ideal_x = parent_bb[1] + parent_bb[3]
    ideal_y = parent_bb[0]
    cost_x = np.abs(ideal_x - child_bb[1]) / parent_bb[3]
    cost_y = np.abs(ideal_y - child_bb[0]) / parent_bb[2]

    # cost of scale difference
    cost_scale = np.abs(parent_bb[2] - child_bb[2]) / parent_bb[2]

    # combined costs
    cost_pair = cost_x + cost_y + cost_scale
    cost_unary = 1 - child_bb[4]
    return  cost_pair * alpha + cost_unary * (1 - alpha)
