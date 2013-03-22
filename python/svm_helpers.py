
import sys
import settings
sys.path.append(settings.libsvm_path)
import svmutil as svm
import numpy as np
import random

def SplitIntoK(k, L):
    chunk_size = int(np.ceil(len(L) / float(k)))
    out_L = []
    for i in range(k):
        i1 = i*chunk_size
        i2 = min((i+1)*chunk_size, len(L))
        out_L.append(L[i1:i2])
    return out_L

from itertools import chain

def CrossValidate(Y, X, param, k_folds=5):
    rand_idx = range(len(Y))
    random.shuffle(rand_idx)
    idx_groups = SplitIntoK(k_folds, rand_idx)
    pos_acc = 0
    neg_acc = 0
    for i in range(k_folds):
        test_idx = idx_groups[i]
        exclude_test = [idx_groups[j] for j in range(len(idx_groups)) if i != j]
        train_idx = list(chain(*exclude_test))

        Y_test = [Y[test_i] for test_i in test_idx]
        X_test = [X[test_i] for test_i in test_idx]        

        Y_train = [Y[train_i] for train_i in train_idx]
        X_train = [X[train_i] for train_i in train_idx]        

        # recompute accuracy
        prob = svm.svm_problem(Y_train,X_train)
        svm_model = svm.svm_train(prob, param)

        p_labs, p_acc, p_vals = svm.svm_predict(Y_test, X_test, svm_model, '-q')

        tps = sum([1 for j in range(len(p_labs)) if (p_labs[j]==1 and Y_test[j]==1)])
        fns = sum([1 for j in range(len(p_labs)) if (p_labs[j]==-1 and Y_test[j]==1)])

        tns = sum([1 for j in range(len(p_labs)) if (p_labs[j]==-1 and Y_test[j]==-1)])
        fps = sum([1 for j in range(len(p_labs)) if (p_labs[j]==1 and Y_test[j]==-1)])

        pos_acc += tps / float(tps + fns)
        neg_acc += tns / float(tns + fps)

    pos_acc = pos_acc / k_folds
    neg_acc = neg_acc / k_folds
    return (pos_acc, neg_acc)
    
def TrainSvmLinear2(Y, X, sweep_c=range(-2,18)):
    num_positives = float(Y.count(1))
    num_negatives = float(Y.count(-1))

    best_c = -1
    best_acc = -1
    for c_pow in sweep_c:
        current_c = np.power(2.0,c_pow)
        param = svm.svm_parameter('-t 0 -c %f -w-1 %f -w1 %f -q' % (current_c,
                                                                    100/num_negatives,
                                                                    100/num_positives))
        current_pos_acc, current_neg_acc = CrossValidate(Y, X, param)
        current_acc = current_pos_acc
        print '%f, %f, %f' % (current_c, current_acc, current_neg_acc)
        if best_acc < current_acc:
            best_acc = current_acc
            best_c = current_c

    prob = svm.svm_problem(Y,X)
    param = svm.svm_parameter('-t 0 -c %f -w-1 %f -w1 %f -q' % (best_c,
                                                                100/num_negatives,
                                                                100/num_positives))
    svm_model = svm.svm_train(prob, param)
    p_labs, p_acc, p_vals = svm.svm_predict(Y, X, svm_model, '-q')
    return svm_model
    
def TrainSvmLinear(Y, X, sweep_c=range(-2,8)):
    num_positives = float(Y.count(1))
    num_negatives = float(Y.count(-1))

    best_c = -1
    best_acc = -1
    for c_pow in sweep_c:
        current_c = np.power(2.0,c_pow)
        prob = svm.svm_problem(Y,X)
        param = svm.svm_parameter('-v 5 -t 0 -c %f -w-1 %f -w1 %f -q' % (current_c,
                                                                         100/num_negatives,
                                                                         100/num_positives))
        current_acc = svm.svm_train(prob, param)
        print '%f, %f' % (current_c, current_acc)
        if best_acc < current_acc:
            best_acc = current_acc
            best_c = current_c

        # recompute accuracy
        param = svm.svm_parameter('-t 0 -c %f -w-1 %f -w1 %f -q' % (best_c,
                                                                    100/num_negatives,
                                                                    100/num_positives))
        svm_model = svm.svm_train(prob, param)
        p_labs, p_acc, p_vals = svm.svm_predict(Y, X, svm_model, '-q')


    prob = svm.svm_problem(Y,X)
    param = svm.svm_parameter('-t 0 -c %f -w-1 %f -w1 %f -q' % (best_c,
                                                                100/num_negatives,
                                                                100/num_positives))
    svm_model = svm.svm_train(prob, param)
    p_labs, p_acc, p_vals = svm.svm_predict(Y, X, svm_model, '-q')
    pdb.set_trace()
    return svm_model

def TrainSvmPoly2(Y, X, sweep_c=range(-2,18)):
    num_positives = float(Y.count(1))
    num_negatives = float(Y.count(-1))

    best_c = -1
    best_acc = -1
    for c_pow in sweep_c:
        current_c = np.power(2.0,c_pow)
        param = svm.svm_parameter('-t 1 -d 2 -c %f -w-1 %f -w1 %f -q' % (current_c,
                                                                              100/num_negatives,
                                                                              100/num_positives))
        current_pos_acc, current_neg_acc = CrossValidate(Y, X, param)
        current_acc = current_pos_acc
        print '%f, %f, %f' % (current_c, current_acc, current_neg_acc)
        if best_acc < current_acc:
            best_acc = current_acc
            best_c = current_c

    prob = svm.svm_problem(Y,X)
    param = svm.svm_parameter('-t 1 -d 2 -c %f -w-1 %f -w1 %f -q' % (best_c, 100/num_negatives,
                                                                     100/num_positives))
    svm_model = svm.svm_train(prob, param)
    p_labs, p_acc, p_vals = svm.svm_predict(Y, X, svm_model, '-q')
    return svm_model

def TrainSvmPoly(Y, X, sweep_c=range(-5,5)):
    num_positives = float(Y.count(1))
    num_negatives = float(Y.count(-1))

    best_c = -1
    best_acc = -1
    for c_pow in sweep_c:
        current_c = np.power(2.0,c_pow)
        prob = svm.svm_problem(Y,X)
        param = svm.svm_parameter('-v 5 -t 1 -d 2 -c %f -w-1 %f -w1 %f -q' % (current_c,
                                                                              100/num_negatives,
                                                                              100/num_positives))
        current_acc = svm.svm_train(prob, param)
        print 'c = %f, cv acc = %f' % (current_c, current_acc)
        if best_acc < current_acc:
            best_acc = current_acc
            best_c = current_c

    prob = svm.svm_problem(Y,X)
    param = svm.svm_parameter('-t 1 -d 2 -c %f -w-1 %f -w1 %f -q' % (best_c, 100/num_negatives,
                                                                     100/num_positives))
    svm_model = svm.svm_train(prob, param)
    p_labs, p_acc, p_vals = svm.svm_predict(Y, X, svm_model, '-q')
    pdb.set_trace()
    return svm_model

def TrainSvmRbf(Y, X, sweep_c=range(-5,5), sweep_g=range(-5,5)):
    num_negatives = float(Y.count(-1))
    num_positives = float(Y.count(1))

    best_c = -1
    best_g = -1
    best_acc = -1
    for c_pow in sweep_c:
        for g_pow in sweep_g:
            current_c = np.power(2.0,c_pow)
            current_g = np.power(2.0,g_pow)
            prob = svm.svm_problem(Y,X)
            param = svm.svm_parameter('-v 5 -t 2 -c %f -g %f -w-1 %f -w1 %f -q' % (current_c,
                                                                                   current_g,
                                                                                   100/num_negatives,
                                                                                   100/num_positives))
            current_acc = svm.svm_train(prob, param)
            print 'c = %f, g = %f, cv acc = %f' % (current_c, current_g, current_acc)
            if best_acc < current_acc:
                best_acc = current_acc
                best_c = current_c
                best_g = current_g

    prob = svm.svm_problem(Y,X)
    param = svm.svm_parameter('-t 2 -c %f -g %f -w-1 %f -w1 %f -q' % (best_c, best_g,
                                                                      100/num_negatives,
                                                                      100/num_positives))
    svm_model = svm.svm_train(prob, param)
    p_labs, p_acc, p_vals = svm.svm_predict(Y, X, svm_model, '-q')
    pdb.set_trace()
    return svm_model

def TrainSvmRbf2(Y, X, sweep_c=range(-5,5), sweep_g=range(-5,5)):
    num_negatives = float(Y.count(-1))
    num_positives = float(Y.count(1))

    best_c = -1
    best_g = -1
    best_acc = -1
    for c_pow in sweep_c:
        for g_pow in sweep_g:
            current_c = np.power(2.0,c_pow)
            current_g = np.power(2.0,g_pow)
            prob = svm.svm_problem(Y,X)
            param = svm.svm_parameter('-t 2 -c %f -g %f -w-1 %f -w1 %f -q' % (current_c,
                                                                              current_g,
                                                                              100/num_negatives,
                                                                              100/num_positives))
            current_pos_acc, current_neg_acc = CrossValidate(Y, X, param)
            current_acc = current_pos_acc
            print 'c = %f, g = %f, cv acc = %f, neg acc = %f' % (current_c, current_g, current_acc,
                                                                 current_neg_acc)
            if best_acc < current_acc:
                best_acc = current_acc
                best_c = current_c
                best_g = current_g

    prob = svm.svm_problem(Y,X)
    param = svm.svm_parameter('-t 2 -c %f -g %f -w-1 %f -w1 %f -q' % (best_c, best_g,
                                                                      100/num_negatives,
                                                                      100/num_positives))
    svm_model = svm.svm_train(prob, param)
    p_labs, p_acc, p_vals = svm.svm_predict(Y, X, svm_model, '-q')
    pdb.set_trace()
    return svm_model


def UpdateWordsWithSvm(svm_model, word_results):
    svm_clf = svm_model[0]
    min_vals = svm_model[1][0]
    max_vals = svm_model[1][1]    

    if not word_results:
        return
    
    X_list = []
    n_features = -1
    for i in range(len(word_results)):
        word_result = word_results[i]
        char_bbs = word_result[1]
        word_score = word_result[0][0,4]
        features = ComputeWordFeatures(char_bbs, word_score)
        if n_features < 0:
            n_features = len(features)
        X_list.append(features)
        
    assert n_features > 0

    X_mat = np.vstack(X_list)
    X_mat = X_mat - min_vals
    X_mat = X_mat / max_vals
    X = [dict(zip(range(n_features), x_i)) for x_i in X_mat.tolist()]
    p_labs, p_acc, p_vals = svm.svm_predict([0]*len(X), X, svm_clf, '-q')
    labels = svm_clf.get_labels()

    for i in range(len(word_results)):
        word_result = word_results[i]
        if labels[0] < 0:
            word_result[0][0,4] = -p_vals[i][0]
        else:
            word_result[0][0,4] = p_vals[i][0]

def ComputeWordFeatures(char_bbs, word_score):
    # 1. features based on raw scores
    #    - median/mean of character scores
    #    - min/max of character scores
    feature_vector = np.array(word_score)
    feature_vector = np.append(feature_vector, np.median(char_bbs[:,4]))
    feature_vector = np.append(feature_vector, np.mean(char_bbs[:,4]))
    feature_vector = np.append(feature_vector, np.min(char_bbs[:,4]))
    feature_vector = np.append(feature_vector, np.max(char_bbs[:,4]))
    feature_vector = np.append(feature_vector, np.std(char_bbs[:,4]))

    # 2. features based on length of string
    feature_vector = np.append(feature_vector, char_bbs.shape[0])

    # 3. features based on global layout
    #    - min/max of horizontal/vertical spaces
    x_diffs = char_bbs[1::,1] - char_bbs[0:-1,1]
    mean_width = np.mean(char_bbs[:,3])
    y_diffs = char_bbs[:,0] - np.mean(char_bbs[:,0])
    mean_height = np.mean(char_bbs[:,2])
    # standard deviation of horizontal gaps
    feature_vector = np.append(feature_vector, np.std(x_diffs / mean_width))
    # standard deviation of vertical variation
    feature_vector = np.append(feature_vector, np.std(y_diffs / mean_height))
    # max horizontal gap
    feature_vector = np.append(feature_vector, np.max(x_diffs / mean_width))
    # max vertical gap
    feature_vector = np.append(feature_vector, np.max(y_diffs / mean_height))

    # 4. features based on scale variation
    mean_scale = np.mean(char_bbs[:,2])
    feature_vector = np.append(feature_vector, np.std(char_bbs[:,2] / mean_scale))
    feature_vector = np.append(feature_vector, np.max(char_bbs[:,2] / mean_scale))

    # TODO: pairwise features?
    return feature_vector

