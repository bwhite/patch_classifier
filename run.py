import glob
import cv2
import vidfeat
import numpy as np
import os
import time
import imfeat
import random
import sklearn
import shutil
import multiprocessing
import contextlib

HOG = imfeat.HOGLatent(sbin=4, blocks=1)


@contextlib.contextmanager
def timer(name):
    st = time.time()
    yield
    print('[%s]: %s' % (name, time.time() - st))


def compute_hog(path):
    return HOG(cv2.imread(path))


def sample_boxes(sz, density=1., sizes=[32, 64, 128]):
    """
    Args:
        sz: Size (width, height)
        num_boxes: Number of boxes
        size: Box size

    Returns:
        Numpy array of shape num_boxes x 4 where each is tl_y, tl_x, br_y, br_x
    """
    for size in sizes:
        sz = np.asarray(sz, dtype=np.int)
        num_boxes = int(np.prod(sz / float(size) * density))
        print('sz:%s num_boxes:%s size:%s' % (sz, num_boxes, size))
        if num_boxes <= 0:
            continue
        try:
            tls = np.dstack([np.random.randint(0, sz[0] - size, num_boxes),
                             np.random.randint(0, sz[1] - size, num_boxes)])[0]
        except ValueError:
            return np.array([]).reshape((0, 4))
        return np.hstack([tls, tls + np.array([size, size])])


def fdr(c, poss, negs):
    p = [c.decision_function(x)[0][0] for x in poss]
    n = [c.decision_function(x)[0][0] for x in negs]
    d = np.mean(p) - np.mean(n)
    d *= d
    return d / (np.var(p) + np.var(n))


def num_pos_score(c, poss, negs):
    p = np.array([c.decision_function(x)[0][0] for x in poss])
    max_n = np.max([c.decision_function(x)[0][0] for x in negs])
    return np.sum(p > max_n)


def num_pos_k_score(c, poss, negs, k=100):
    p = np.array([c.decision_function(x)[0][0] for x in poss])
    n = np.array([c.decision_function(x)[0][0] for x in negs])
    return np.sum(np.argsort(np.hstack([p, n]))[-k:] < p.size)


def ndcg_score_old(c, poss, negs, k=25):
    p = np.array([c.decision_function(x)[0][0] for x in poss])
    n = np.array([c.decision_function(x)[0][0] for x in negs])
    rels = np.argsort(np.hstack([p, n]))[::-1][:k] < p.size
    print('|p|=%d |n|=%d sample=%s' % (p.size, n.size, rels[:20]))
    irels = sorted(rels, reverse=True)
    dcg = lambda r: np.sum([float(y) / np.log2(x + 3) for x, y in enumerate(r)])
    return np.nan_to_num(dcg(rels) / dcg(irels))


def ndcg_score(p, n, k):
    p = np.asfarray(p)
    n = np.asfarray(n)
    rels = np.argsort(np.hstack([p, n]))[::-1][:k] < p.size
    print('|p|=%d |n|=%d sample=%s' % (p.size, n.size, rels[:20]))
    irels = sorted(rels, reverse=True)
    dcg = lambda r: np.sum([float(y) / np.log2(x + 3) for x, y in enumerate(r)])
    return np.nan_to_num(dcg(rels) / dcg(irels))


def load_features(path):
    pos_feats = []
    val_feats = []
    neg_feats = []
    neg_paths = glob.glob(path + '/0/*.png')
    random.shuffle(neg_paths)
    val_paths = neg_paths[500:1000]
    neg_paths = neg_paths[:500]
    # Neg
    for x in neg_paths:
        neg_feats.append(compute_hog(x))
    neg_feats = np.asfarray(neg_feats)
    # Val
    for x in val_paths:
        val_feats.append(compute_hog(x))
    val_feats = np.asfarray(val_feats)
    # Pos
    pos_paths = glob.glob(path + '/1/*.png')
    random.shuffle(pos_paths)
    pos_paths = pos_paths[:501]
    for x in pos_paths:
        pos_feats.append(compute_hog(x))
    pos_feats = np.asfarray(pos_feats)
    return pos_feats, val_feats, neg_feats, pos_paths, val_paths, neg_paths


def compute_paths(path, num_pos_train=1000, num_neg_train=5000, num_pos_test=1000, num_neg_test=1000):
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


def _compute_validation_scores(pos_test_paths, neg_test_paths, cs, ps=None, ns=None, k=25):
    if ps is None:
        ps = [[] for x in cs]
    if ns is None:
        ns = [[] for x in cs]
    print('|cs|:%d |pos_test_paths|:%d |neg_test_paths|:%d' % (len(cs), len(pos_test_paths), len(neg_test_paths)))
    for f in compute_features(pos_test_paths):
        for p, c in zip(ps, cs):
            p.append(c.decision_function(f)[0][0])
    for f in compute_features(neg_test_paths):
        for n, c in zip(ns, cs):
            n.append(c.decision_function(f)[0][0])
    return [ndcg_score(p, n, k) for n, p in zip(ns, ps)], ps, ns


def identify_descriminative_patches(path):
    # In the patch path, train an exemplar SVM classifier for each patch and
    # keep the patches that best separate the pos/neg patches
    pos_train_paths, neg_train_paths, pos_test_paths, neg_test_paths = compute_paths(path)
    with timer('Negative Training Features'):
        neg_train_feats = np.asfarray(list(compute_features(neg_train_paths)))
    num_iters = 5
    cs = []
    c_paths = list(pos_train_paths)
    # Make training matrix
    neg_feats_one_pos = np.vstack([np.zeros_like(neg_train_feats[0]), neg_train_feats])
    labels = np.hstack([np.ones(1), np.zeros(len(neg_train_feats))])
    with timer('Positive Training Features and Train Exemplars'):
        for pos_ind, pos_feat in enumerate(compute_features(pos_train_paths)):
            neg_feats_one_pos[0, :] = pos_feat
            cs.append(train_exemplar(neg_feats_one_pos, labels))
    prev_pos_ind = 0
    prev_neg_ind = 0
    ns = ps = None
    for x in range(num_iters):
        num_pos_test = (x + 1) * len(pos_test_paths) / num_iters
        num_neg_test = (x + 1) * len(neg_test_paths) / num_iters
        num_exemplars = len(pos_train_paths) / 2 ** (x + 1)
        print('num_pos_test:%d num_neg_test:%d num_exemplars:%d' % (num_pos_test, num_neg_test, num_exemplars))
        try:
            os.makedirs('exemplars/%.3d/worst/' % x)
            os.makedirs('exemplars/%.3d/best/' % x)
        except OSError:
            pass
        with timer('compute_validation_scores'):
            scores, ps, ns = _compute_validation_scores(pos_test_paths[prev_pos_ind:num_pos_test],
                                                        neg_test_paths[prev_neg_ind:num_neg_test],
                                                        cs, ps, ns)
        prev_pos_ind = num_pos_test
        prev_neg_ind = num_neg_test
        k = min(100, len(scores) / 2)
        scores_inds = np.argsort(scores)
        for y in range(k):
            shutil.copy(c_paths[scores_inds[y]], 'exemplars/%.3d/worst/%.3d-%f.png' % (x, y, scores[scores_inds[y]]))
            shutil.copy(c_paths[scores_inds[-(y + 1)]], 'exemplars/%.3d/best/%.3d-%f.png' % (x, y, scores[scores_inds[-(y + 1)]]))
        cs, c_paths, ps, ns = zip(*[(cs[z], c_paths[z], ps[z], ns[z])
                                    for z in scores_inds[-num_exemplars:]])


def single_exemplar(path, exemplar_path):
    """Outputs ranked samples for a single exemplar"""
    pos_feat = HOG(cv2.imread(exemplar_path))
    pos_feats, val_feats, neg_feats, pos_paths, val_paths, neg_paths = load_features(path)
    neg_feats_one_pos = np.vstack([pos_feat, neg_feats])
    labels = np.hstack([np.ones(1), np.zeros(len(neg_feats))])
    c = train_exemplar(neg_feats_one_pos, labels)

    pos_scores = [c.decision_function(x)[0][0] for x in pos_feats]
    val_scores = [c.decision_function(x)[0][0] for x in val_feats]
    pos_inds = np.argsort(pos_scores)
    val_inds = np.argsort(val_scores)

    try:
        os.makedirs('exemplars/pos')
        os.makedirs('exemplars/neg')
    except OSError:
        pass
    output_coeff(c)
    for x in range(100):
        shutil.copy(pos_paths[pos_inds[x]], 'exemplars/pos/worst-%.2d-%f.png' % (x, pos_scores[pos_inds[x]]))
        shutil.copy(pos_paths[pos_inds[-(x + 1)]], 'exemplars/pos/best-%.2d-%f.png' % (x, pos_scores[pos_inds[-(x + 1)]]))
        shutil.copy(val_paths[val_inds[x]], 'exemplars/neg/worst-%.2d-%f.png' % (x, val_scores[val_inds[x]]))
        shutil.copy(val_paths[val_inds[-(x + 1)]], 'exemplars/neg/best-%.2d-%f.png' % (x, val_scores[val_inds[-(x + 1)]]))



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
    

def cluster_images(path):
    pos_feats = []
    pos_paths = glob.glob(path + '/1/*.png')
    random.shuffle(pos_paths)
    for x in pos_paths[:1000]:
        pos_feats.append(HOG(cv2.imread(x)))
    pos_feats = np.asfarray(pos_feats)
    import scipy as sp
    import scipy.cluster
    #pos_feats = sp.cluster.vq.whiten(pos_feats)
    for pos_path, index in zip(pos_paths, sp.cluster.vq.kmeans2(pos_feats, 20, minit='points')[1]):
        d = 'clusters/%d/' % index
        try:
            os.makedirs(d)
        except OSError:
            pass
        shutil.copy(pos_path, d + os.path.basename(pos_path))


def save_boxes(path, frame, boxes):
    try:
        os.makedirs(path)
    except OSError:
        pass
    st = time.time()
    for x, y in enumerate(boxes):
        cur_path = '%s/%f-%.5d.png' % (path, st, x)
        patch = cv2.resize(frame[y[0]:y[2], y[1]:y[3]], (32, 32))
        cv2.imwrite(cur_path, patch)


def main():
    #max_side = 320
    for label, frame in vidfeat.load_label_frames('/home/brandyn/playground/aladdin_data_cropped/person/'):
        #sz = np.array(frame.shape[:2])
        #frame = cv2.resize(frame, tuple(np.array(max_side * sz / np.max(sz), dtype=np.int)[::-1]))
        boxes = sample_boxes(frame.shape[:2])
        save_boxes('boxes/%d' % label, frame, boxes)
        print(label)


def process():
    identify_descriminative_patches('boxes')

if __name__ == '__main__':
    POOL = multiprocessing.Pool()
    #single_exemplar('boxes', 'mx.png')
    #main()
    process()
    #cluster_images('boxes')
