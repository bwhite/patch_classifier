import random
import os
import glob
import shutil
import cv2
import numpy as np
import scipy as sp
import scipy.cluster
from sklearn.cluster import DBSCAN
import cPickle as pickle


def cluster_images(path, hog):
    pos_feats = []
    pos_paths = glob.glob(path + '/1/*.png')
    random.shuffle(pos_paths)
    for x in pos_paths[:1000]:
        pos_feats.append(hog(cv2.imread(x)))
    pos_feats = np.asfarray(pos_feats)
    #pos_feats = sp.cluster.vq.whiten(pos_feats)
    for pos_path, index in zip(pos_paths, sp.cluster.vq.kmeans2(pos_feats, 20, minit='points')[1]):
        d = 'clusters/%d/' % index
        try:
            os.makedirs(d)
        except OSError:
            pass
        shutil.copy(pos_path, d + os.path.basename(pos_path))


def collect_ids(t):
    out = []
    if t.left is not None:
        out += collect_ids(t.left)
    if t.right is not None:
        out += collect_ids(t.right)
    if t.left is None and t.right is None:
        out.append(t.id)
    return out
    

def thresh_tree(t, c):
    if c > 0:
        out = []
        if t.left is not None:
            out += thresh_tree(t.left, c - 1)
        print('l:%s r:%s d:%s i:%s c:%s' % (t.left, t.right, t.dist, t.id, t.count))
        if t.right is not None:
            out += thresh_tree(t.right, c - 1)
        if t.left is None and t.right is None:
            out.append([t.id])
        return out
    else:
        return [collect_ids(t)]


def cluster_exemplars(c_paths, s):
    #ps = np.array(ps)
    #s = 2 - (sp.stats.spearmanr(ps, axis=1)[0] + 1)
    #pickle.dump((c_paths, s), open('out3.pkl', 'w'), -1)
    #print s[0]
    db = DBSCAN(eps=.5, metric='precomputed', min_samples=5).fit(s)
    print(db.core_sample_indices_)
    clusts = db.labels_
    print(clusts)
    pickle.dump(db, open('db.pkl', 'w'), -1)
    #return
    #s = sp.spatial.distance.squareform(s)
    #l = sp.cluster.hierarchy.complete(s)
    #clusts = scipy.cluster.hierarchy.fcluster(l, len(c_paths) / 4, criterion='maxclust')
    l = {}
    for x, y in enumerate(clusts):
        l.setdefault(y, []).append(x)
    for cluster_num, nodes in l.items():
        d = 'clusters/%.5d-%d/' % (cluster_num, len(nodes))
        os.makedirs(d)
        for node in nodes:
            shutil.copy(c_paths[node], d + '%d.png' % node)
