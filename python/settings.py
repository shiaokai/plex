import numpy as np
import cv2
import pdb

hog = cv2.HOGDescriptor((16,16),(16,16),(16,16),(8,8),9,1,-1)
canon_size = (48, 48)

alphabet_master='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_'
alphabet_detect='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
alphabet_upper='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

char_train_dir = '/data/text/plex/icdar/train/charHard'
char_test_dir = '/data/text/plex/icdar/test/charHard'
char_train_bg_dir = '/data/text/plex/msrc/train/charBg'
char_test_bg_dir = '/data/text/plex/msrc/test/charBg'

img_train_dir = '/data/text/plex/icdar/train/images'
img_test_dir = '/data/text/plex/icdar/test/images'

img_train_gt_dir = '/data/text/plex/icdar/train/wordAnn'
img_test_gt_dir = '/data/text/plex/icdar/test/wordAnn'

lex0_train_dir = '/data/text/plex/icdar/train/lex0'
lex0_test_dir = '/data/text/plex/icdar/test/lex0'

bootstrap_img_dir = '/data/text/plex/msrc/train/images'
bootstrap_bg_dir = '/data/text/plex/icdarBt_py/train/charBg'
initial_char_clf_name = 'char_clf_initial.dat'
char_clf_name = 'char_clf_final.dat'

max_per_class = np.inf
max_bg = 10000

n_procs = 6

detect_idxs=[]
for alpha in alphabet_detect:
    detect_idxs.append(alphabet_master.find(alpha))

cache_dir = 'cache'

case_mapping = []
for i in range(len(alphabet_detect)):
    case_mapping.append(alphabet_upper.find(alphabet_detect[i].upper()))
    
