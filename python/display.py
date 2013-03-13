import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pdb

def OutputCharBbs(img, bbs, alphabet, output_dir='dbg'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(len(alphabet)):
        DrawCharBbs(img, bbs, alphabet, filter_label=i)
        # store image
        plt.savefig(os.path.join(output_dir,"charDet_%s.png" % (alphabet[i])))

def DrawCharBbs(img, bbs, alphabet, filter_label=-1, draw_top=-1):
    fig = plt.figure()
    plt.cla()
    ax = fig.add_subplot(111)
    img_rgb = img[:,:,[2,1,0]]
    plt.imshow(img_rgb);
    if draw_top>0:
        # sort by score
        sidx = np.argsort(bbs[:,4])
        sidx = sidx[::-1]
        for i in range(draw_top):
            if i > bbs.shape[0]:
                break
            else:
                bb = bbs[sidx[i],:]
                patch = mpl.patches.Rectangle((bb[1],bb[0]),
                                              bb[3],bb[2],
                                              color='green',
                                              fill=False)
                # draw rectangle
                ax.add_patch(patch)
                plt.text(bb[1],bb[0],"%s:%.02f" % (alphabet[int(bb[5])],float(bb[4])),
                         backgroundcolor=(1,1,1))
                # draw text
                
    else:
        for i in range(bbs.shape[0]):
            bb = bbs[i,:]
            if filter_label>=0 and bb[5] != filter_label:
                continue
            else:
                # draw me
                patch = mpl.patches.Rectangle((bb[1],bb[0]),
                                              bb[3],bb[2],
                                              color='green',
                                              fill=False)
                ax.add_patch(patch)
                plt.text(bb[1],bb[0],"%s:%.02f" % (alphabet[int(bb[5])],float(bb[4])),
                         backgroundcolor=(1,1,1))


def DrawEvalResults(img, gt_result, dt_result, show_error_text=False):
    fig = plt.figure(1)
    plt.cla()
    ax = fig.add_subplot(111)
    img_rgb = img[:,:,[2,1,0]]
    plt.imshow(img_rgb);

    # draw false negatives
    for gt_result0 in gt_result:
        if gt_result0[1]:
            continue
        bb = gt_result0[2]
        patch = mpl.patches.Rectangle((bb[1],bb[0]),
                                      bb[3],bb[2],
                                      color='m',
                                      fill=False)
        ax.add_patch(patch)
        if show_error_text:
            plt.text(bb[1],bb[0],"FN:%s" % (gt_result0[0]), backgroundcolor=(1,1,1))
        

    # draw false positives
    for dt_result0 in dt_result:
        if dt_result0[1]:
            continue
        bb = dt_result0[2]
        patch = mpl.patches.Rectangle((bb[1],bb[0]),
                                      bb[3],bb[2],
                                      color='r',
                                      fill=False)
        ax.add_patch(patch)
        if show_error_text:
            plt.text(bb[1],bb[0],"FP:%s%1.2f" % (dt_result0[0], dt_result0[3]),
                     backgroundcolor=(1,1,1))
        
    # draw true positives
    for dt_result0 in dt_result:
        if not dt_result0[1]:
            continue
        char_bbs = dt_result0[4]
        for char_bb in char_bbs:
            char_patch = mpl.patches.Rectangle((char_bb[1],char_bb[0]),
                                               char_bb[3],char_bb[2],
                                               color='g', fill=False,
                                               linestyle='dashed')
            ax.add_patch(char_patch)
        
        word_bb = dt_result0[2]
        word_patch = mpl.patches.Rectangle((word_bb[1],word_bb[0]),
                                      word_bb[3],word_bb[2],
                                      color='g', fill=False)
        ax.add_patch(word_patch)
        plt.text(word_bb[1],word_bb[0],"TP:%s%1.2f" % (dt_result0[0], dt_result0[3]),
                 backgroundcolor=(1,1,1))

