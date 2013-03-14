#/usr/bin/env python

import numpy as np
import cv,cv2
import pdb
from common import clock, draw_str
from video import create_capture
from matplotlib.pyplot import show, imshow, figure, title
from hog_utils import draw_hog2

help_message = '''
USAGE: peopledetect.py <image_names> ...

Press any key to continue, ESC to stop.
'''

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

def look_at_hog(img):
    dims=(16,16)
    t = clock()
    hog = cv2.HOGDescriptor(dims,(16,16),(16,16),(8,8),9,1,-1)
    feature_vector = hog.compute(img, winStride=dims, padding=(0,0))
    dt = clock() - t
    print 'Extraction took: %.1f ms' % (dt*1000)
    t = clock()
    hog_viz=draw_hog2(img,hog,feature_vector)
    dt = clock() - t
    print 'drawing took: %.1f ms' % (dt*1000)
    return hog_viz

if __name__ == '__main__':
    import sys
    from glob import glob
    import itertools as it

    # -- temporary dimension mismatch
    if len(sys.argv)==2:
        img = cv2.imread(sys.argv[1])
        hog_viz=look_at_hog(img)
        imshow(hog_viz,cmap='gray');show()
    else:
        hog = cv2.HOGDescriptor()
        #hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
        cam = create_capture(1, fallback='synth:bg=../cpp/lena.jpg:noise=0.05')
        while True:
            ret, img = cam.read()
            vis = img.copy()
            t = clock()
            if 1:
                vis=look_at_hog(vis)
            else:
                found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
                found_filtered = []
                for ri, r in enumerate(found):
                    for qi, q in enumerate(found):
                        if ri != qi and inside(r, q):
                            break
                        else:
                            found_filtered.append(r)
                            draw_detections(vis, found)
                            draw_detections(vis, found_filtered, 3)
                print '%d (%d) found.' % (len(found_filtered), len(found))
                            
            dt = clock() - t
            draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
            cv2.imshow('img', vis)
            cv2.waitKey(500)
            if 0xFF & cv2.waitKey(5) == 27:
                break

        cv2.destroyAllWindows()




