import glob
import cv2
import numpy as np
import os
import time
import imfeat
import random
import sklearn
import shutil
import multiprocessing
import contextlib
import cPickle as pickle
from data import write_boxes

HOG = imfeat.HOGLatent(sbin=4, blocks=1)


@contextlib.contextmanager
def timer(name):
    st = time.time()
    yield
    print('[%s]: %s' % (name, time.time() - st))


def compute_hog(path):
    return HOG(cv2.imread(path))


def ndcg_score(p, n, k):
    p = np.asfarray(p)
    n = np.asfarray(n)
    rels = np.argsort(np.hstack([p, n]))[::-1][:k] < p.size
    print('|p|=%d |n|=%d sample=%s' % (p.size, n.size, rels[:20]))
    irels = sorted(rels, reverse=True)
    dcg = lambda r: np.sum([float(y) / np.log2(x + 3) for x, y in enumerate(r)])
    return np.nan_to_num(dcg(rels) / dcg(irels))


class BadExemplar(Exception):
    """The exemplar cannot meet the required performance characteristics"""


def precision_threshold(p, n, target=.9, k=200):
    p = np.asfarray(p)
    n = np.asfarray(n)
    gts = (np.argsort(np.hstack([p, n]))[::-1][:k] < p.size).astype(np.int)
    confs = np.sort(np.hstack([p, n]))[::-1][:k]
    p, r, t = sklearn.metrics.precision_recall_curve(gts, confs)
    p = p[:-1]
    r = r[:-1]
    if target > np.max(p):
        print('Warning: BadExemplar max(p) = %f' % np.max(p))
        raise BadExemplar
    pinds = (p >= target).nonzero()[0]
    ind = pinds[np.argmax(r[pinds])]
    precision, recall, threshold = p[ind], r[ind], t[ind]
    preds = (confs >= threshold).astype(np.int)
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
    precision_check = tp / float(tp + fp)
    recall_check = tp / float(tp + fn)
    print((precision, precision_check))
    print((recall, recall_check))
    assert precision_check == precision
    assert recall_check == recall
    print('p:%f r:%f t:%f' % (precision, recall, threshold))
    return threshold


def calibrate_exemplars(ps, ns):
    for num, (p, n) in enumerate(zip(ps, ns)):
        try:
            yield precision_threshold(p, n), num
        except BadExemplar:
            pass


def compute_paths(path, num_pos_train=2000, num_neg_train=50000, num_pos_test=10000, num_neg_test=10000):
    neg_paths = glob.glob(path + '/0/*.png')
    random.shuffle(neg_paths)
    neg_train_paths = neg_paths[:num_neg_train]
    neg_test_paths = neg_paths[num_neg_train:][:num_neg_test]
    pos_paths = glob.glob(path + '/1/*.png')
    random.shuffle(pos_paths)
    pos_train_paths = pos_paths[:num_pos_train]
    pos_test_paths = pos_paths[num_pos_train:][:num_pos_test]
    return pos_train_paths, neg_train_paths, pos_test_paths, neg_test_paths


def compute_features(paths):
    #for x in paths:
    #    yield compute_hog(x)
    return POOL.imap(compute_hog, paths)


def compute_features_single(paths):
    for x in paths:
        yield compute_hog(x)


def train_classifiers(data):
    pos_train_paths, neg_feats_one_pos, labels = data
    cs = []
    for pos_ind, pos_feat in enumerate(compute_features_single(pos_train_paths)):
        neg_feats_one_pos[0, :] = pos_feat
        cs.append(train_exemplar(neg_feats_one_pos, labels))
    return cs


def compute_decision_funcs(data):
    test_paths, cs = data
    outs = [[] for x in cs]
    for f in compute_features_single(test_paths):
        for o, c in zip(outs, cs):
            o.append(c.decision_function(f)[0][0])
    return outs


def _compute_validation_scores(pos_test_paths, neg_test_paths, cs, ps=None, ns=None, k=200):
    if ps is None:
        ps = [[] for x in cs]
    if ns is None:
        ns = [[] for x in cs]
    print('|cs|:%d |pos_test_paths|:%d |neg_test_paths|:%d' % (len(cs), len(pos_test_paths), len(neg_test_paths)))
    for ps_new in POOL.imap(compute_decision_funcs, ((x, cs) for x in chunks(pos_test_paths))):
        for p, p_new in zip(ps, ps_new):
            p += p_new
    for ns_new in POOL.imap(compute_decision_funcs, ((x, cs) for x in chunks(neg_test_paths))):
        for n, n_new in zip(ns, ns_new):
            n += n_new
    return [ndcg_score(p, n, k) for n, p in zip(ns, ps)], ps, ns


def chunks(l):
    """ Yield successive n-sized chunks from l."""
    import math
    n = int(math.ceil(len(l) / 32.))
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def identify_descriminative_patches(path, exemplar_path=None):
    pos_train_paths, neg_train_paths, pos_test_paths, neg_test_paths = compute_paths(path)
    if exemplar_path:
        pos_train_paths = [exemplar_path]
    with timer('Negative Training Features'):
        neg_train_feats = np.asfarray(list(compute_features(neg_train_paths)))
    cs = []
    c_paths = list(pos_train_paths)
    # Make training matrix
    neg_feats_one_pos = np.vstack([np.zeros_like(neg_train_feats[0]), neg_train_feats])
    labels = np.hstack([np.ones(1), np.zeros(len(neg_train_feats))])
    with timer('Positive Training Features and Train Exemplars'):
        cs = sum(POOL.imap(train_classifiers, ((x, neg_feats_one_pos, labels)
                                               for x in chunks(pos_train_paths))), [])
    prev_pos_ind = 0
    prev_neg_ind = 0
    ns = ps = None
    #(1000, 1000), (2000, 2000), (4000, 4000),
    for run_iter, (num_pos_test, num_neg_test) in enumerate([(len(pos_test_paths), len(neg_test_paths))]):
        #num_pos_test = (x + 1) * len(pos_test_paths) / num_iters
        #num_neg_test = (x + 1) * len(neg_test_paths) / num_iters
        #num_exemplars = len(pos_train_paths)  # / 2 ** (run_iter + 1)
        print('num_pos_test:%d num_neg_test:%d' % (num_pos_test, num_neg_test))
        try:
            os.makedirs('exemplars/%.3d/worst/' % run_iter)
            os.makedirs('exemplars/%.3d/best/' % run_iter)
        except OSError:
            pass
        with timer('compute_validation_scores'):
            scores, ps, ns = _compute_validation_scores(pos_test_paths[prev_pos_ind:num_pos_test],
                                                        neg_test_paths[prev_neg_ind:num_neg_test],
                                                        cs, ps, ns)
            ts, inds = zip(*calibrate_exemplars(ps, ns))
            scores, cs, c_paths = zip(*[(scores[x], cs[x], c_paths[x]) for x in inds])
        prev_pos_ind = num_pos_test
        prev_neg_ind = num_neg_test
        k = max(1, min(200, len(scores) / 2))
        scores_inds = np.argsort(scores)
        for y in range(k):
            shutil.copy(c_paths[scores_inds[y]], 'exemplars/%.3d/worst/%.3d-%f.png' % (run_iter, y, scores[scores_inds[y]]))
            shutil.copy(c_paths[scores_inds[-(y + 1)]], 'exemplars/%.3d/best/%.3d-%f.png' % (run_iter, y, scores[scores_inds[-(y + 1)]]))
        cs, c_paths, ps, ns = zip(*[(cs[z], c_paths[z], ps[z], ns[z])
                                    for z in scores_inds])
    return cs, c_paths, pos_test_paths, neg_test_paths, ps, ns, ts


def single_exemplar(path, exemplar_path):
    cs, c_paths, pos_test_paths, neg_test_paths, ps, ns = identify_descriminative_patches(path, exemplar_path)
    pos_scores = ps[0]
    neg_scores = ns[0]
    pos_inds = np.argsort(pos_scores)
    neg_inds = np.argsort(neg_scores)
    try:
        os.makedirs('exemplars/pos')
        os.makedirs('exemplars/neg')
    except OSError:
        pass
    output_coeff(cs[0])
    for x in range(200):
        shutil.copy(pos_test_paths[pos_inds[x]], 'exemplars/pos/worst-%.3d-%f.png' % (x, pos_scores[pos_inds[x]]))
        shutil.copy(pos_test_paths[pos_inds[-(x + 1)]], 'exemplars/pos/best-%.3d-%f.png' % (x, pos_scores[pos_inds[-(x + 1)]]))
        shutil.copy(neg_test_paths[neg_inds[x]], 'exemplars/neg/worst-%.3d-%f.png' % (x, neg_scores[neg_inds[x]]))
        shutil.copy(neg_test_paths[neg_inds[-(x + 1)]], 'exemplars/neg/best-%.3d-%f.png' % (x, neg_scores[neg_inds[-(x + 1)]]))


def output_coeff(c):
    d = int(np.sqrt(c.coef_.size / 31))
    masks = c.coef_.reshape((d, d, 31))
    t = 0
    for x in range(masks.shape[2]):
        m = np.min(masks[:, :, x])
        M = np.max(masks[:, :, x])
        norm_mask = (masks[:, :, x] - m) / (M - m + .0000001)
        t += norm_mask
        out = np.array(255 * norm_mask, dtype=np.uint8)
        cv2.imwrite('exemplars/%.2d.png' % x, out)
    cv2.imwrite('exemplars/%.2d.png' % (x + 1), np.array(255 * t / masks.shape[2], dtype=np.uint8))


def train_exemplar(features, labels):
    c = sklearn.svm.LinearSVC(verbose=10, C=10**20)
    return c.fit(features, labels)


def sliding_hog(exemplars, image, feature, sbin=4, size=32, cells=6):
    exemplar_coefs = np.ascontiguousarray([x.coef_.ravel() for x in exemplars])
    exemplar_intercepts = np.ascontiguousarray([x.intercept_[0] for x in exemplars])
    print(exemplar_intercepts.shape)
    print(exemplar_coefs.shape)
    #d = lambda b: np.dot(c.coef_, b) + c.intercept_
    with timer('Hog full'):
        f = feature(image, ravel=False)
    with timer('Slide/Predict Everywhere'):
        runs = 0
        for x in range(f.shape[0] - cells):
            for y in range(f.shape[0] - cells):
                fs = np.ascontiguousarray(f[x:x + cells, y:y + cells, :].ravel())
                (np.dot(exemplar_coefs, fs) + exemplar_intercepts)
                runs += 1
        print('Runs:%d' % runs)

    #f = feature(cv2.resize(image, (, 32)), ravel=False)
    print(np.sum(f, axis=2).shape)
    f = feature(image[:35, :35, :], ravel=False)[:6, :6, :]
    print(np.sum(f, axis=2))

    return f
    


if __name__ == '__main__':
    POOL = multiprocessing.Pool()
    #single_exemplar('boxes', 'mx.png')
    #write_boxes
    out = identify_descriminative_patches('boxes')
    #with open('out.pkl', 'w') as fp:
    #    pickle.dump(out, fp, -1)
    #cluster_images('boxes')
    #with open('out.pkl') as fp:
    #    cs, c_paths, pos_test_paths, neg_test_paths, ps, ns = pickle.load(fp)
    #cs = []
    #f = sliding_hog(cs, cv2.imread('/home/brandyn/playground/aladdin_data_cropped/person/0/00000008.jpg'), HOG)
    #    c_paths, s = pickle.load(fp)
    #cluster_exemplars(c_paths, s)
