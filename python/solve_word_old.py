import numpy as np
import pdb
from helpers import UnionBbs
from nms_old import WordBbsNms

def ComputePairScore(parent_bb, child_bb, alpha):
    if child_bb[1] < parent_bb[1]:
        # child cannot be to the left of parent
        return np.inf
    
    # costs of x and y offsets
    ideal_x = parent_bb[1] + parent_bb[3]
    ideal_y = parent_bb[0]
    cost_x = np.abs(ideal_x - child_bb[1]) / parent_bb[3]
    cost_y = np.abs(ideal_y - child_bb[0]) / parent_bb[2]

    # cost of scale difference
    cost_scale = np.abs(parent_bb[2] - child_bb[2]) / parent_bb[2]

    # combined costs
    cost_pair = cost_x + 2 * cost_y + cost_scale
    cost_unary = 1 - child_bb[4]
    return  cost_pair * alpha + cost_unary * (1 - alpha)

def SolveWord(bbs, word, alphabet, max_locations, alpha, overlap_thr):
    # HACK: check that every letter in word exists in bbs
    for i in range(len(word)):
        cur_char = word[i]
        idx = alphabet.find(word[i])
        if np.sum(bbs[:,5]==idx) == 0:
            return []
        
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
        parent_char = word[i_proc-1]

        child_idx  = alphabet.find(child_char)
        child_bbs  = bbs[bbs[:,5]==child_idx,:]

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
                    score = ComputePairScore(parent_bb, child_bb, alpha)
                    if best_child_score > score:
                        best_child_score = score
                        best_child_idx = k

                # store min score and index
                dp_costs_j[j] = best_child_score
                dp_ptrs_j[j] = best_child_idx
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
                    score = ComputePairScore(parent_bb, child_bb, alpha) + dp_costs_child[k]
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
            
    root_scores = np.zeros(num_roots)
    
    best_root_score = np.inf
    best_root_idx = -1

    for j in range(num_roots):
        root_bb = root_bbs[j,:]
        score = (1 - root_bb[4]) * (1-alpha) + dp_costs_root[j]
        root_scores[j] = score
        if best_root_score > score:
            best_root_score = score
            best_root_idx = j

    dp_ptrs.reverse()
    all_word_results = []
    sorted_idx = np.argsort(root_scores)
    # collect all configurations that end in non inf roots
    for i in range(num_roots):
        cur_bbs_idx = [i]
        cur_root_score = root_scores[i]
        if cur_root_score == np.inf:
            continue
        
        for j in range(len(word)-1):
            dp_ptrs_j = dp_ptrs[j]
            cur_bbs_idx.append(dp_ptrs_j[cur_bbs_idx[j]])
    
        cur_bbs = np.zeros((len(word),6))
        for j in range(len(word)):
            bb_idx = cur_bbs_idx[j]
            char_idx  = alphabet.find(word[j])
            char_bbs = bbs[bbs[:,5]==char_idx]
            if (char_bbs.ndim == 1):
                cur_bbs[j,:] = char_bbs
            else:
                cur_bbs[j,:] = char_bbs[bb_idx,:]

        adjusted_score = - (cur_root_score / len(word))
        word_bb = np.append(UnionBbs(cur_bbs), adjusted_score)
        all_word_results.append([word_bb, cur_bbs])

    # perform word-level NMS
    word_results = WordBbsNms(all_word_results, overlap_thr=overlap_thr)
    word_results = word_results[0:min(len(word_results), max_locations)]

    return word_results

