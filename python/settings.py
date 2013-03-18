import numpy as np
import cv2
import os

libsvm_path='/home/shiaokai/projects/third_party/libsvm/libsvm-3.16/python/'

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

lex5_train_dir = '/data/text/plex/icdar/train/lex5'
lex5_test_dir = '/data/text/plex/icdar/test/lex5'

lex20_train_dir = '/data/text/plex/icdar/train/lex20'
lex20_test_dir = '/data/text/plex/icdar/test/lex20'

bootstrap_img_dir = '/data/text/plex/msrc/train/images'
bootstrap_bg_dir = '/data/text/plex/icdarBt_py/train/charBg'
initial_char_clf_name = 'char_clf_initial.dat'

project_dir = '/home/shiaokai/projects/github/plex/python'
char_clf_name = os.path.join(project_dir, 'char_clf_final.dat')
word_clf_name = os.path.join(project_dir, 'word_clf.dat')

max_per_class = np.inf
max_bg = 10000

n_procs = 6

# char params
overlap_thr=0.5
score_thr=0.1
min_height=0.05
min_pixel_height=10

fig_dir='/home/shiaokai/Dropbox'

# word params
max_locations = 5

alpha=.15

detect_idxs=[]
for alpha0 in alphabet_detect:
    detect_idxs.append(alphabet_master.find(alpha0))

cache_dir = 'cache'

case_mapping = []
for i in range(len(alphabet_detect)):
    case_mapping.append(alphabet_upper.find(alphabet_detect[i].upper()))
    
