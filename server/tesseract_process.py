# read STDIN and call tesseract on each bounding box
import pdb
import sys
import os
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

orig_img = sys.argv[1]
out_img = sys.argv[2]

img = cv2.imread(orig_img)

fig = plt.figure()
ax = fig.add_subplot(111)
img_rgb = img[:,:,[2,1,0]]
plt.imshow(img_rgb)

f = sys.stdin
lines = f.readlines()
for line in lines:
    parts = line.split(' ')
    if parts[0] == 'total':
        continue
    x = int(parts[0])
    y = int(parts[1])
    w = int(parts[2])
    h = int(parts[3])
    # todo add 25% padding
    patch = mpl.patches.Rectangle((x,y), w, h,
                                  color='green',
                                  fill=False)
    ax.add_patch(patch)
    img_patch = img_rgb[y:y+h,x:x+w,:]
    cv2.imwrite('temp_patch.png', img_patch)
    # call tesseract
    cmd = "tesseract temp_patch.png tess_out"
    os.system(cmd)
    # read output
    tesseract_output = []
    with open('tess_out.txt', 'r') as f:
        tesseract_output += f.readlines()

    tess_cat = filter(str.isalnum, (''.join(tesseract_output)).strip())
    plt.text(x, y, tess_cat, backgroundcolor=(1,1,1))
                
plt.savefig(out_img)
