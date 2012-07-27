import hadoopy
import numpy as np
import cPickle as pickle
import os
import shutil
from random_uniform import LeakyPriorityQueue
import glob
import re


def _find_exemplar_fn(exemplar_path, exemplar):
    image_id, box, = exemplar[:2]
    fns = glob.glob(exemplar_path + '/*')
    out, = [fn for fn in fns if fn.endswith('-%s-%s.png' % (image_id, box))]
    return out


def main():
    path = 'exemplarbank/output/1341790878.92/val_pred_pos_kern2'
    exemplars = pickle.load(open('exemplars.pkl'))
    exemplar_path = 'exemplars'
    exemplar_ids = {}
    for y, x in enumerate(exemplars):
        x = x[0][0]
        exemplar_ids.setdefault(x, []).append(y)
    try:
        shutil.rmtree('hik_pairs')
    except OSError:
        pass
    os.makedirs('hik_pairs')
    pq = LeakyPriorityQueue(100)
    for (kernel, row_num), columns in hadoopy.readtb(path):
        if kernel != 'hik':
            continue
        print(row_num)
        # Blacklist all exemplars from the same image
        columns[exemplar_ids[exemplars[row_num][0][0]]] = -np.inf
        for column_num, val in enumerate(columns[:row_num]):
            pq.add(-val, (row_num, column_num))
    for num, (score, (row_num, max_col)) in enumerate(pq.items_sorted()):
        shutil.copy(_find_exemplar_fn(exemplar_path, exemplars[row_num][0]), 'hik_pairs/%.5d-a-%f.png' % (num, -score))
        shutil.copy(_find_exemplar_fn(exemplar_path, exemplars[max_col][0]), 'hik_pairs/%.5d-b-%f.png' % (num, -score))


def main2():
    exemplar_name = 'e05c099586f744a6d9e70b334e79da08-[0.5217391304347826, 0.0, 0.8695652173913043, 0.9523809523809523]'
    path = 'exemplarbank/output/1341790878.92/val_pred_pos_kern2'
    exemplars = pickle.load(open('exemplars.pkl'))
    exemplar_path = 'exemplars'
    exemplar_ids = {}
    exemplar_num = None
    for exemplar_num, ((image_id, box, _), _) in enumerate(exemplars):
        if exemplar_name == '%s-%s' % (image_id, box):
            break
    for y, x in enumerate(exemplars):
        x = x[0][0]
        exemplar_ids.setdefault(x, []).append(y)
    try:
        shutil.rmtree('hik_pairs_specific')
    except OSError:
        pass
    os.makedirs('hik_pairs_specific')
    pq = LeakyPriorityQueue(100)
    for (kernel, row_num), columns in hadoopy.readtb(path):
        if kernel != 'hik' or row_num != exemplar_num:
            continue
        print(row_num)
        # Blacklist all exemplars from the same image
        columns[exemplar_ids[exemplars[row_num][0][0]]] = -np.inf
        for column_num, val in enumerate(columns[:row_num]):
            pq.add(-val, (row_num, column_num))
    for num, (score, (row_num, max_col)) in enumerate(pq.items_sorted()):
        shutil.copy(_find_exemplar_fn(exemplar_path, exemplars[row_num][0]), 'hik_pairs_specific/%.5d-a-%f.png' % (num, -score))
        shutil.copy(_find_exemplar_fn(exemplar_path, exemplars[max_col][0]), 'hik_pairs_specific/%.5d-b-%f.png' % (num, -score))


if __name__ == '__main__':
    main2()
