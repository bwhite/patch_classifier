import hadoopy
import numpy as np
import cPickle as pickle
import os
import shutil


def _find_exemplar_fn(exemplar_path, exemplar):
    image_id, box, score = exemplar
    return os.path.join(exemplar_path, 'exemplar-%.5d-%s-%s.png' % (score, image_id, box))
    

def main():
    path = 'exemplarbank/output/1341790878.92/val_pred_pos_kern2'
    exemplars = pickle.load(open('exemplars.pkl'))
    exemplar_path = 'exemplars'
    try:
        shutil.rmtree('hik_pairs')
    except OSError:
        pass
    os.makedirs('hik_pairs')
    pairs = []
    max_vals = {}
    for (kernel, row_num), columns in hadoopy.readtb(path):
        if kernel != 'hik':
            continue
        #print(row_num)
        columns[row_num] = -np.inf
        max_col = np.argmax(columns)
        try:
            max_vals[max_col] += 1
        except KeyError:
            max_vals[max_col] = 1
        pairs.append((columns[max_col], row_num, max_col))
    print(max_vals)
    pairs.sort(reverse=True)
    for num, (score, row_num, max_col) in enumerate(pairs):
        shutil.copy(_find_exemplar_fn(exemplar_path, exemplars[row_num][0]), 'hik_pairs/%.5d-a.png' % num)
        shutil.copy(_find_exemplar_fn(exemplar_path, exemplars[max_col][0]), 'hik_pairs/%.5d-b.png' % num)

if __name__ == '__main__':
    main()
