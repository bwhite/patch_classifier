import hadoopy
import cPickle as pickle
import numpy as np
import scipy as sp
import scipy.cluster
from pair_hik import _find_exemplar_fn
import os
import shutil


def main():
    exemplar_feats = list(hadoopy.readtb('exemplarbank/output/1341790878.92/pos_sample'))
    feats = np.vstack([x[1] for x in exemplar_feats])
    print(feats.shape)
    try:
        shutil.rmtree('clusters')
    except OSError:
        pass
    os.makedirs('clusters')
    for exemplar_num, cluster_num in enumerate(sp.cluster.vq.kmeans2(feats, 20, minit='points')[1]):
        fn = _find_exemplar_fn('exemplars', exemplar_feats[exemplar_num][0])
        cluster_path = 'clusters/%d/' % cluster_num
        try:
            os.makedirs(cluster_path)
        except OSError:
            pass
        shutil.copy(fn, cluster_path)

main()
