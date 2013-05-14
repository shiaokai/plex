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
    plt.axis('off')
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


def DebugCharBbs(img, char_bbs, alphabet, lexicon):
    img_rgb = img[:,:,[2,1,0]]
    active_letters = str(''.join(lexicon)).upper()
    active_letters = ''.join(set(sorted(active_letters)))

    fig = plt.figure(figsize=(20*len(active_letters),40))
    plt.cla()
    plt.axis('off')
    counter = 1

    for active_letter in active_letters:
        letter_idx = alphabet.find(active_letter)
        bbs = char_bbs[char_bbs[:,5]==letter_idx,:]
        ax = fig.add_subplot(len(active_letters), 1, counter)
        counter += 1
        plt.imshow(img_rgb);
        for i in range(bbs.shape[0]):
            bb = bbs[i,:]
            patch = mpl.patches.Rectangle((bb[1],bb[0]),
                                          bb[3],bb[2],
                                          color='green',
                                          fill=False)
            ax.add_patch(patch)
            plt.text(bb[1],bb[0],"%s:%.02f" % (alphabet[int(bb[5])],float(bb[4])),
                     backgroundcolor=(1,1,1), size=6)

def DrawWordResults2(img, word_results, score_threshold=-np.inf,
                     show_char_bbs=True):
    """
    Give each word its own figure
    """
    all_words = set([word_result[2] for word_result in word_results])
    n_subplots = len(all_words) + 1
    fig = plt.figure(figsize=(10, 10*n_subplots))
    plt.cla()

    img_rgb = img[:,:,[2,1,0]]

    # draw all bounding boxes
    word_results.reverse()
    for word_result in word_results:
        ax = fig.add_subplot(n_subplots, 1, 1)
        plt.imshow(img_rgb);
        word_bb = word_result[0][0]
        patch = mpl.patches.Rectangle((word_bb[1],word_bb[0]),
                                      word_bb[3],word_bb[2],
                                      color='green',
                                      fill=False)
        ax.add_patch(patch)
        plt.text(word_bb[1],word_bb[0],"%s:%.02f" % (word_result[2], float(word_bb[4])),
                 backgroundcolor=(1,1,1))

    # draw individual word results
    counter = 2
    for active_word in all_words:
        ax = fig.add_subplot(n_subplots, 1, counter)
        plt.imshow(img_rgb);
        counter += 1
        for word_result in word_results:
            if word_result[2] != active_word:
                continue

            word_bb = word_result[0][0]
            patch = mpl.patches.Rectangle((word_bb[1],word_bb[0]),
                                          word_bb[3],word_bb[2],
                                          color='green',
                                          fill=False)
            ax.add_patch(patch)
            plt.text(word_bb[1],word_bb[0],"%s:%.02f" % (word_result[2], float(word_bb[4])),
                     backgroundcolor=(1,1,1))

            if show_char_bbs:
                char_bbs = word_result[1]
                for char_bb in char_bbs:
                    char_patch = mpl.patches.Rectangle((char_bb[1],char_bb[0]),
                                                       char_bb[3],char_bb[2],
                                                       color='c', fill=False,
                                                       linestyle='dashed')
                    ax.add_patch(char_patch)

def DrawWordResults(img, word_results, score_threshold=-np.inf,
                    show_char_bbs=False):
    fig = plt.figure()
    plt.cla()
    plt.axis('off')
    ax = fig.add_subplot(111)
    img_rgb = img[:,:,[2,1,0]]
    plt.imshow(img_rgb);

    for word_result in word_results:
        word_bb = word_result[0][0]
        if word_bb[4] < score_threshold:
            continue
        patch = mpl.patches.Rectangle((word_bb[1],word_bb[0]),
                                      word_bb[3],word_bb[2],
                                      color='green',
                                      fill=False)
        ax.add_patch(patch)
        plt.text(word_bb[1],word_bb[0],"%s:%.02f" % (word_result[2], float(word_bb[4])),
                 backgroundcolor=(1,1,1))

        if show_char_bbs:
            char_bbs = word_result[1]
            for char_bb in char_bbs:
                char_patch = mpl.patches.Rectangle((char_bb[1],char_bb[0]),
                                                   char_bb[3],char_bb[2],
                                                   color='c', fill=False,
                                                   linestyle='dashed')
                ax.add_patch(char_patch)

def DrawEvalResults(img, gt_result, dt_result, show_error_text=False):
    fig = plt.figure(1)
    plt.cla()
    plt.axis('off')
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

