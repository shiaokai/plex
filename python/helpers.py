import settings
import numpy as np
import hashlib
import os

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

#def CollapseLetterCase(bbs):
'''
Set the bbs to their capitals
'''
