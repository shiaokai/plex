import pdb
import os, sys
import settings
import cPickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import hashlib

from char_det import CharDetector
from display import OutputCharBbs, DrawCharBbs
from helpers import GetCachePath
from wordspot import WordSpot

if len(sys.argv) is 4:
    img_name = sys.argv[1]
    lexicon = [sys.argv[2]]
    max_locations = int(sys.argv[3])
elif len(sys.argv) is 1:
    img_name = 'data/scales.JPG'
    lexicon = ['ANS']
    max_locations = 3
else:
    print 'Usage: $ python demo_wordspot.py [image_path] [string] [instances]'
    sys.exit(0)

img = cv2.imread(img_name)
(match_bbs, char_bbs) = WordSpot(img, lexicon, use_cache=False,
                                 img_name=img_name,
                                 max_locations=max_locations,
                                 alpha=.15,
                                 overlap_thr=0.25)

word_bbs = np.zeros((len(match_bbs), 6))
for i in range(len(match_bbs)):
    word_bbs[i,:] = match_bbs[i][0]

# draw match
DrawCharBbs(img, word_bbs, settings.alphabet_master)
plt.show()

#OutputCharBbs(img, char_bbs, settings.alphabet_master)
