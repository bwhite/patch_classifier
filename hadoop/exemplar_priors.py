import hadoopy
from pair_hik import _find_exemplar_fn
import cv2
import numpy as np
import os
import shutil
import cPickle as pickle


def main():
    path = 'exemplarbank/output/1341790878.92/val_pred_pos'
    pyramid, num_boxes = hadoopy.readtb(path).next()[1]
    try:
        shutil.rmtree('priors')
    except OSError:
        pass
    os.makedirs('priors')
    exemplars = pickle.load(open('exemplars.pkl'))
    for exemplar_num in range(pyramid.shape[0]):
        print(exemplar_num)
        p = pyramid[exemplar_num, :, :] / float(np.max(pyramid[exemplar_num, :, :]))
        p = (p * 255).astype(np.uint8)
        print p
        cv2.imwrite('priors/%.5d-%.5d.png' % (exemplars[exemplar_num][0][2], exemplar_num), p)


if __name__ == '__main__':
    main()
