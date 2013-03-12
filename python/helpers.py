import settings
import numpy as np
import hashlib
import os
import pdb
import cv2

def ReadAllImages(char_dir, bg_dir, char_classes,
                  max_per_class=np.inf, max_bg=np.inf):
    '''
    This function follows up to one directory down
    '''
    # walk directory and load in n images
    imgs=np.zeros(0)
    labels=np.zeros(0)
    k = 0
    max_allocate=5e4
    for class_index in range(len(char_classes)):
        cur_class = char_classes[class_index]
        if cur_class == '_':
            imgs_dir = bg_dir  # background class
        else:
            if cur_class.islower():
                cur_class='-'+cur_class  # lowercase have minus
            imgs_dir = os.path.join(char_dir,cur_class)

        for root, dirs, files in os.walk(imgs_dir):
            # add all the files within dirs to the files list
            for cur_dir in dirs:
                for root1, dirs1, files1 in os.walk(os.path.join(imgs_dir,cur_dir)):
                    if not files1:
                        break
                    files1_with_parents = [cur_dir + os.sep + f for f in files1]
                    files += files1_with_parents
                    
            if cur_class == '_':
                if (max_bg < np.inf) and (len(files) > max_bg):
                    random.shuffle(files)
                    files = files[0:max_bg]

            else:
                if (max_per_class < np.inf) and (len(files) > max_per_class):
                    random.shuffle(files)
                    files = files[0:max_per_class]


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

def CollapseLetterCase(bbs, mapping):
    '''
    Set the bbs to their capitals
    '''
    for i in range(bbs.shape[0]):
        bbs[i,5] = mapping[int(bbs[i,5])]

    return bbs

