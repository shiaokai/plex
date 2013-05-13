import numpy as np
import cv2
import os
from cfg_train_synth_test_icdar import *
#from cfg_train_icdar_test_icdar import *

libsvm_path='/home/shiaokai/projects/third_party/libsvm/libsvm-3.16/python/'
swt_dir='/home/shiaokai/projects/github/ccv/bin/'

hog = cv2.HOGDescriptor((16,16),(16,16),(16,16),(8,8),9,1,-1)
canon_size = (48, 48)

alphabet_master='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_'
alphabet_detect='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
alphabet_upper='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

char_train_bg_dir = '/data/text/plex/msrc/train/charBg'
char_test_bg_dir = '/data/text/plex/msrc/test/charBg'
bootstrap_img_dir = '/data/text/plex/msrc/train/images'

project_dir = '/home/shiaokai/projects/github/plex/python'
stored_dir = os.path.join(project_dir, 'data_' + base_name)
if not(os.path.isdir(stored_dir)):
    os.makedirs(stored_dir)
    
initial_char_clf_name = os.path.join(stored_dir, 'char_clf_initial.dat')
char_clf_name = os.path.join(stored_dir, 'char_clf_final.dat')
word_clf_name = os.path.join(stored_dir, 'word_clf.dat')
word_clf_meta_name = os.path.join(stored_dir, 'word_clf_meta.dat')
word_clf_poly_name = os.path.join(stored_dir, 'word_clf_poly.dat')

n_procs = 6

# char params
overlap_thr=0.5
score_thr=0.1
min_height=0.05
min_pixel_height=10

fig_dir = os.path.join('/home/shiaokai/Dropbox', base_name)
if not(os.path.isdir(fig_dir)):
    os.makedirs(fig_dir)

# word params
max_locations = 5

detect_idxs=[]
for alpha0 in alphabet_detect:
    detect_idxs.append(alphabet_master.find(alpha0))

cache_dir = 'cache'

case_mapping = []
for i in range(len(alphabet_detect)):
    case_mapping.append(alphabet_upper.find(alphabet_detect[i].upper()))
    
