import settings
import numpy as np
import hashlib
import os
import pdb
import cv2
import random
import string

def ValidateString(input):
    # return code 1 is valid, 0 is not
    filtered_input = [s if settings.alphabet_detect.find(s) > -1 else '' for s in input]
    filtered_string = string.join(filtered_input,sep='')
    filtered_string = filtered_string.upper()
    if len(filtered_string) > 2:
        return (1, filtered_string)
    else:
        return (0, filtered_string)

def ReadAllImages(char_dir, bg_dir, char_classes,
                  max_per_class=np.inf, max_bg=np.inf):
    '''
    This function follows up to one directory down
    '''
    # walk directory and load in n images
    imgs=np.zeros(0)
    labels=np.zeros(0)
    k = 0
    max_allocate=1e5
    for class_index, cur_class in enumerate(char_classes):
        if cur_class == '_':
            imgs_dir = bg_dir  # background class
        else:
            if cur_class.islower():
                cur_class='-'+cur_class  # lowercase have minus
            imgs_dir = os.path.join(char_dir,cur_class)

        filelist = []
        for root, dirs, files in os.walk(imgs_dir):
            add_files = [os.path.join(root, f) for f in files]
            filelist += add_files
            if cur_class == '_':
                if (max_bg < np.inf) and (len(files) > max_bg):
                    random.shuffle(filelist)
                    filelist = filelist[0:max_bg]
            else:
                if (max_per_class < np.inf) and (len(filelist) > max_per_class):
                    random.shuffle(filelist)
                    filelist = filelist[0:max_per_class]

        for path in filelist:
            p1,ext=os.path.splitext(path)
            if ext!='.png':
                continue
            I = cv2.imread(path)
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

    print 'Loaded %i images' % k
    imgs=imgs[:,:,:,0:k]
    return (imgs,labels)

def GetCachePath(path):
    m = hashlib.md5()
    m.update(path)
    if not(os.path.isdir(settings.cache_dir)):
        os.makedirs(settings.cache_dir)
    
    return os.path.join(settings.cache_dir, m.hexdigest())

def UnionBbs(bbs):
    '''
    Return superset of bounding boxes
    '''
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

def BbsOverlap(bb1, bb2):
    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]

    bb1_start_y = bb1[0]
    bb1_start_x = bb1[1]
    bb2_start_y = bb2[0]
    bb2_start_x = bb2[1]

    bb1_end_y = bb1[0] + bb1[2]
    bb1_end_x = bb1[1] + bb1[3]
    bb2_end_y = bb2[0] + bb2[2]
    bb2_end_x = bb2[1] + bb2[3]

    """
    intersect divided by union
    """
    intersect_width = min(bb1_end_x, bb2_end_x) - max(bb1_start_x, bb2_start_x)
    if intersect_width <= 0:
        return 0
    intersect_height = min(bb1_end_y, bb2_end_y) - max(bb1_start_y, bb2_start_y)
    if intersect_width <= 0:
        return 0
    intersect_area = intersect_width * intersect_height
    union_area = bb1_area + bb2_area - intersect_area
    overlap = intersect_area / union_area
    return overlap

def CollapseLetterCase(bbs, mapping):
    '''
    Set the bbs to their capitals
    '''
    for i in range(bbs.shape[0]):
        bbs[i,5] = mapping[int(bbs[i,5])]

    return bbs

