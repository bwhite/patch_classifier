import hadoopy
import os
import numpy as np
import sklearn
import alggen


class Mapper(object):

    def __init__(self):
        pass

    def map(self, key, value):
        input_file = os.environ['map_input_file'].rsplit('/', 2)[1]
        if input_file.find('exemplars') != -1:
            yield key, (0, value)
        elif input_file.find('val_pos') != -1:
            yield key, (1, value)
        elif input_file.find('val_neg') != -1:
            yield key, (2, value)
        else:
            raise ValueError('Unknown input')


class BadExemplar(Exception):
    """The exemplar cannot meet the required performance characteristics"""


def compute_fpr_tpr(preds, gts):
    tp = fp = tn = fn = 0
    for pred, gt in zip(preds, gts):
        if gt:
            if pred:
                tp +=1
            else:
                fn += 1
        else:
            if pred:
                fp += 1
            else:
                tn += 1
    
    fpr = fp / float(fp + tn)
    tpr = tp / float(tp + fn)
    return fpr, tpr


def fpr_threshold(p, n, target=.0001, k=200):
    p = np.asfarray(p)
    n = np.asfarray(n)
    print('|p|=%d |n|=%d' % (p.size, n.size))
    gts = np.ascontiguousarray((np.argsort(np.hstack([p, n]))[::-1] < p.size).astype(np.int))
    confs = np.ascontiguousarray(np.sort(np.hstack([p, n]))[::-1])
    fprs, tprs, threshs = map(np.ascontiguousarray, zip(*alggen.confuse_to_roc_points(alggen.gts_confs_to_cms(gts, confs))))
    if target < np.min(fprs):
        print('Warning: BadExemplar min(fprs) = %f' % np.min(fprs))
        raise BadExemplar
    #print('fprs: %s' % fprs)
    #print('tprs: %s' % tprs)
    #print('threshs: %s' % threshs)
    pinds = (fprs <= target).nonzero()[0]
    ind = pinds[np.argmax(tprs[pinds])]
    fpr, tpr, thresh = fprs[ind], tprs[ind], threshs[ind]
    if 1:
        preds = (confs >= thresh).astype(np.int)
        fpr_check, tpr_check = compute_fpr_tpr(preds, gts)
        if fpr_check != fpr or tpr_check != tpr:
            raise ValueError('FPR/TPR incorrect[%s][%s][%s][%s]' % (fpr_check, fpr, tpr_check, tpr))
    score = np.sum(gts[:k])
    print('fpr:%f tpr:%f thresh:%f score@%d:%d' % (fpr, tpr, thresh, k, score))
    return thresh, score


class Reducer(object):

    def __init__(self):
        pass

    def reduce(self, key, values):
        # Setup data
        # TODO(brandyn): Use multi-file join pattern
        data = [None, None, None]
        for input_type, value in values:
            data[input_type] = value
        if len([x for x in data if x is None]) != 0:
            raise ValueError('Reducer did not get all necessary parts!')
        exemplar, pos, neg = data
        # Compute threshold and output new exemplar
        try:
            thresh, score = fpr_threshold(pos, neg)
        except BadExemplar:
            print('Bad exemplar[%s]' % (key,))
            return
        print('Good exemplar[%s][%f]' % (key, thresh))
        key[2] = score
        yield key, (exemplar[0], exemplar[1] - thresh)
        

if __name__ == '__main__':
    hadoopy.run(Mapper, Reducer)
