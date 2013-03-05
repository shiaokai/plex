import numpy as np
import cv2
import pdb

hog = cv2.HOGDescriptor((16,16),(16,16),(16,16),(8,8),9,1,-1)
canon_size = (48, 48)
alphabet_master='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
alphabet_detect='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
char_train_dir = '/data/text/plex/icdar/train/charHard'
char_test_dir = '/data/text/plex/icdar/test/charHard'
char_train_bg_dir = '/data/text/plex/msrc/train/charBg'
char_test_bg_dir = '/data/text/plex/msrc/test/charBg'
bootstrap_img_dir = '/data/text/plex/msrc/train/images'
bootstrap_bg_dir = '/data/text/plex/icdarBt_py/train/charBg'
char_clf_name = 'char_clf_boot.dat'

max_per_class = np.inf
max_bg = 1000

detect_idxs=[]
for alpha in alphabet_detect:
    detect_idxs.append(alphabet_master.find(alpha))
